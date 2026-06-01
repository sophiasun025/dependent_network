"""Python translations of the dependent network helper functions.

This module mirrors `Functions/dependent_network_functions.R` closely enough to
support side-by-side validation while the project moves from R to Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from jax import random as jrandom
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import pdist, squareform
from numpy.typing import NDArray
Array = NDArray[Any]



@dataclass
class DependentSBMResult:
    """Container matching the list returned by the original R generator."""

    K1: Array
    K2: Array
    adj_list: list[Array]
    graph_list: list[nx.Graph]
    adj_matrix_flat: Array
    edge_names: list[str]
    E_before: Array
    E_after: Array
    community: Array
    tau: int
    rho: float


def gaussian_kernel(X: Array) -> Array:
    """Gaussian kernel matching the original R `gaussiankernel`.

    R reference:
    `Dx = dist(X)^2; sigma = median(Dx); K = exp(-Dx / sigma / 2) - diag(n)`.
    The median is taken over condensed pairwise squared distances, excluding the
    diagonal, just like `dist(X)^2` in R.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    n = X.shape[0]
    condensed_squared_dist = pdist(X, metric="sqeuclidean")
    sigma = float(np.median(condensed_squared_dist)) if condensed_squared_dist.size else 0.0
    if sigma == 0.0:
        sigma += 0.1

    squared_dist = squareform(condensed_squared_dist)
    return np.exp(-squared_dist / sigma / 2.0) - np.eye(n)


@jax.jit
def _gaussian_kernel_jax_r(X: jax.Array) -> jax.Array:
    n = X.shape[0]
    squared_norms = jnp.sum(X * X, axis=1, keepdims=True)
    squared_dist = squared_norms + squared_norms.T - 2.0 * (X @ X.T)
    squared_dist = jnp.maximum(squared_dist, 0.0)

    off_diag = ~jnp.eye(n, dtype=bool)
    condensed_like = jnp.where(off_diag, squared_dist, jnp.nan)
    sigma = jnp.nanmedian(condensed_like)
    sigma = jnp.where(sigma == 0.0, sigma + 0.1, sigma)
    return jnp.exp(-squared_dist / sigma / 2.0) - jnp.eye(n, dtype=X.dtype)


def gaussian_kernel_jax(X: Array) -> Array:
    """JAX-accelerated Gaussian kernel matching the original R function."""

    X_jax = jnp.asarray(X, dtype=jnp.float64)
    if X_jax.ndim != 2:
        raise ValueError("X must be a 2D array")
    return np.asarray(_gaussian_kernel_jax_r(X_jax))


def _make_probability_matrices_jax(
    p_within_before: float,
    p_across_before: float,
    p_within_after: float,
    p_across_after: float,
    block_nums: Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    block_nums_jax = jnp.asarray(block_nums, dtype=jnp.int32)
    community = jnp.repeat(jnp.arange(1, block_nums_jax.size + 1), block_nums_jax, total_repeat_length=int(np.sum(block_nums)))
    same_community = community[:, None] == community[None, :]
    non_diag = 1.0 - jnp.eye(community.size, dtype=jnp.float64)

    E_before = jnp.where(same_community, p_within_before, p_across_before) * non_diag
    E_after = jnp.where(same_community, p_within_after, p_across_after) * non_diag
    return E_before, E_after, community


@partial(jax.jit, static_argnames=("T",))
def _generate_adjacency_sequence_jax(
    key: jax.Array,
    rho: float,
    E_before: jax.Array,
    E_after: jax.Array,
    tau: int,
    T: int,
) -> jax.Array:
    N = E_before.shape[0]
    upper_mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)

    key, first_key = jrandom.split(key)
    A0_upper = jrandom.bernoulli(first_key, E_before).astype(jnp.int32)
    A0 = jnp.where(upper_mask, A0_upper, 0)
    A0 = A0 + A0.T

    def step(carry: tuple[jax.Array, jax.Array], t: jax.Array) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        A_old, key_old = carry
        key_new, draw_key = jrandom.split(key_old)
        E_current = jnp.where(t < tau, E_before, E_after)
        prob_mat = jnp.where(A_old == 1, rho * (1.0 - E_current) + E_current, (1.0 - rho) * E_current)
        prob_mat = prob_mat * (1.0 - jnp.eye(N, dtype=jnp.float64))
        A_upper = jrandom.bernoulli(draw_key, prob_mat).astype(jnp.int32)
        A_new = jnp.where(upper_mask, A_upper, 0)
        A_new = A_new + A_new.T
        return (A_new, key_new), A_new

    (_, _), rest = lax.scan(step, (A0, key), jnp.arange(1, T))
    return jnp.concatenate([A0[None, :, :], rest], axis=0)


def generate_dependent_sbm_cp_jax(
    rho: float,
    p_within_before: float,
    p_across_before: float,
    p_within_after: float,
    p_across_after: float,
    block_nums: list[int] | Array,
    tau: int,
    T: int,
    seed: int = 0,
    include_graphlet_kernel: bool = True,
) -> DependentSBMResult:
    """JAX-accelerated version of `generate_dependent_sbm_cp`.

    The dynamic SBM simulation and Gaussian kernel are handled with JAX array
    operations. The optional graphlet kernel still uses NetworkX because the
    translated R code depends on graph-level combinatorics that are not a good
    fit for JIT compilation in this form.
    """

    block_nums_np = np.asarray(block_nums, dtype=int)
    if tau < 2 or tau >= T - 2:
        raise ValueError("tau should satisfy 2 <= tau < T - 2, matching the R check")

    E_before_jax, E_after_jax, community_jax = _make_probability_matrices_jax(
        p_within_before,
        p_across_before,
        p_within_after,
        p_across_after,
        block_nums_np,
    )
    adj_stack_jax = _generate_adjacency_sequence_jax(
        jrandom.PRNGKey(seed),
        rho,
        E_before_jax,
        E_after_jax,
        tau,
        T,
    )

    adj_stack = np.asarray(adj_stack_jax)
    N = adj_stack.shape[1]
    upper_idx = np.triu_indices(N, k=1)
    adj_matrix_flat = adj_stack[:, upper_idx[0], upper_idx[1]]
    edge_names = [f"edge_{i + 1}_{j + 1}" for i, j in zip(*upper_idx)]

    K1 = gaussian_kernel_jax(adj_matrix_flat)
    adj_list = [adj_stack[t] for t in range(T)]
    graph_list = [nx.from_numpy_array(A) for A in adj_list]
    if include_graphlet_kernel:
        K2 = calculate_graphlet_kernel(graph_list, size=3)
        np.fill_diagonal(K2, 0.0)
    else:
        K2 = np.zeros((T, T), dtype=float)

    return DependentSBMResult(
        K1=K1,
        K2=K2,
        adj_list=adj_list,
        graph_list=graph_list,
        adj_matrix_flat=adj_matrix_flat,
        edge_names=edge_names,
        E_before=np.asarray(E_before_jax),
        E_after=np.asarray(E_after_jax),
        community=np.asarray(community_jax),
        tau=tau,
        rho=rho,
    )



def _calculate_graphlet_kernel_r(graph_list: list[nx.Graph], size: int = 3) -> Array:
    """Compute graphlet kernel via R graphkernels::CalculateGraphletKernel.

    This is the exact backend used by the original R simulation code. It writes
    temporary adjacency matrices, calls Rscript, and reads the returned kernel.
    """

    with tempfile.TemporaryDirectory(prefix="kap_graphkernels_") as tmp:
        tmp_path = Path(tmp)
        graph_dir = tmp_path / "graphs"
        graph_dir.mkdir()
        output_path = tmp_path / "K2.csv"
        script_path = tmp_path / "calculate_graphlet_kernel.R"

        for idx, graph in enumerate(graph_list, start=1):
            adjacency = nx.to_numpy_array(graph, dtype=int)
            np.savetxt(graph_dir / f"graph_{idx}.csv", adjacency, fmt="%d", delimiter=",")

        script_path.write_text(
            """
drive_r_lib <- Sys.getenv("KAP_R_LIB", "/content/drive/MyDrive/R/colab-library")
if (dir.exists(drive_r_lib)) {
  .libPaths(c(drive_r_lib, .libPaths()))
}
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(graphkernels))
args <- commandArgs(trailingOnly = TRUE)
graph_dir <- args[[1]]
output_path <- args[[2]]
k <- as.integer(args[[3]])
files <- list.files(graph_dir, pattern = "\\\\.csv$", full.names = TRUE)
files <- files[order(as.numeric(gsub("[^0-9]", "", basename(files))))]
graph_list <- lapply(files, function(path) {
  A <- as.matrix(read.csv(path, header = FALSE, check.names = FALSE))
  graph_from_adjacency_matrix(A, mode = "undirected", diag = FALSE)
})
K <- CalculateGraphletKernel(graph_list, k)
write.table(K, file = output_path, sep = ",", row.names = FALSE, col.names = FALSE)
""".lstrip()
        )

        completed = subprocess.run(
            ["Rscript", str(script_path), str(graph_dir), str(output_path), str(size)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "R graphkernels backend failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return np.loadtxt(output_path, delimiter=",", dtype=float)


def _to_grakel_graph(graph: nx.Graph) -> Any:
    from grakel import Graph

    adjacency = nx.to_numpy_array(graph, dtype=int)
    return Graph(adjacency)


def _calculate_graphlet_kernel_grakel(
    graph_list: list[nx.Graph],
    size: int = 3,
    sampling: dict[str, Any] | None = None,
    normalize: bool = False,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> Array:
    from grakel import GraphletSampling

    grakel_graphs = [_to_grakel_graph(graph) for graph in graph_list]
    kernel = GraphletSampling(
        k=size,
        sampling=sampling,
        normalize=normalize,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return np.asarray(kernel.fit_transform(grakel_graphs), dtype=float)


def calculate_graphlet_kernel(
    graph_list: list[nx.Graph],
    size: int = 3,
    backend: str = "r",
    sampling: dict[str, Any] | None = None,
    normalize: bool = False,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> Array:
    """Compute the graphlet kernel.

    By default this calls the R `graphkernels::CalculateGraphletKernel` package
    to match the original simulation exactly. Set `backend="grakel"` only when
    you intentionally want the Python GraKeL implementation instead.
    """

    if backend == "r":
        return _calculate_graphlet_kernel_r(graph_list, size=size)
    if backend == "grakel":
        return _calculate_graphlet_kernel_grakel(
            graph_list,
            size=size,
            sampling=sampling,
            normalize=normalize,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    raise ValueError("backend must be 'r' or 'grakel'")


def generate_dependent_sbm_cp(
    rho: float,
    p_within_before: float,
    p_across_before: float,
    p_within_after: float,
    p_across_after: float,
    block_nums: list[int] | Array,
    tau: int,
    T: int,
    rng: np.random.Generator | None = None,
) -> DependentSBMResult:
    """Generate a dependent stochastic block model with one change point.

    This is the Python analogue of `generate_dependent_sbm_cp` in the R file.
    Time indexing follows the R logic: the initial graph uses the before-change
    probabilities, transitions with t < tau use `E_before`, and later
    transitions use `E_after`.
    """

    rng = np.random.default_rng() if rng is None else rng
    block_nums = np.asarray(block_nums, dtype=int)
    N = int(block_nums.sum())
    community = np.repeat(np.arange(1, len(block_nums) + 1), block_nums)

    if tau < 2 or tau >= T - 2:
        raise ValueError("tau should satisfy 2 <= tau < T - 2, matching the R check")

    def make_E(p_within: float, p_across: float) -> Array:
        E = np.full((N, N), p_across, dtype=float)
        E[community[:, None] == community[None, :]] = p_within
        np.fill_diagonal(E, 0.0)
        return E

    E_before = make_E(p_within_before, p_across_before)
    E_after = make_E(p_within_after, p_across_after)
    upper_idx = np.triu_indices(N, k=1)

    A = np.zeros((N, N), dtype=int)
    A[upper_idx] = rng.binomial(1, E_before[upper_idx])
    A = A + A.T
    adj_list = [A]

    for t in range(1, T):
        A_old = adj_list[t - 1]
        E_current = E_before if t < tau else E_after

        prob_mat = np.zeros((N, N), dtype=float)
        old_edge = A_old == 1
        old_no_edge = A_old == 0
        prob_mat[old_edge] = rho * (1.0 - E_current[old_edge]) + E_current[old_edge]
        prob_mat[old_no_edge] = (1.0 - rho) * E_current[old_no_edge]
        np.fill_diagonal(prob_mat, 0.0)

        A_new = np.zeros((N, N), dtype=int)
        A_new[upper_idx] = rng.binomial(1, prob_mat[upper_idx])
        A_new = A_new + A_new.T
        adj_list.append(A_new)

    adj_matrix_flat = np.vstack([A[upper_idx] for A in adj_list])
    edge_names = [f"edge_{i + 1}_{j + 1}" for i, j in zip(*upper_idx)]

    graph_list = [nx.from_numpy_array(A) for A in adj_list]
    K1 = gaussian_kernel(adj_matrix_flat)
    K2 = calculate_graphlet_kernel(graph_list, size=3)
    np.fill_diagonal(K2, 0.0)

    return DependentSBMResult(
        K1=K1,
        K2=K2,
        adj_list=adj_list,
        graph_list=graph_list,
        adj_matrix_flat=adj_matrix_flat,
        edge_names=edge_names,
        E_before=E_before,
        E_after=E_after,
        community=community,
        tau=tau,
        rho=rho,
    )


def _lag_matrix(n: int) -> Array:
    idx = np.arange(n)
    return np.abs(idx[:, None] - idx[None, :])


def _lag_stat(K_work: Array, lag_mat: Array, reducer: str) -> dict[int, float]:
    stats: dict[int, float] = {}
    for lag in np.unique(lag_mat):
        values = K_work[lag_mat == lag]
        values = values[np.isfinite(values)]
        if values.size == 0:
            stats[int(lag)] = np.nan
        elif reducer == "mean":
            stats[int(lag)] = float(np.mean(values))
        elif reducer == "median":
            stats[int(lag)] = float(np.median(values))
        elif reducer == "sd":
            stats[int(lag)] = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
        else:
            raise ValueError(f"Unknown reducer: {reducer}")
    return stats



def remove_lag_effect(K: Array, max_lag: float = np.inf, s: float = 0.6) -> dict[str, Any]:
    """Remove lag effects from a kernel matrix.

    Returns the same named outputs as the current R function: residual kernels
    adjusted by raw/smoothed lag means and ratio kernels adjusted by raw/smoothed
    lag medians.
    """

    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    K_work = K.copy()
    np.fill_diagonal(K_work, np.nan)
    lag_mat = _lag_matrix(n)

    lag_mean = _lag_stat(K_work, lag_mat, "mean")
    lag_median = _lag_stat(K_work, lag_mat, "median")
    lag_sd = _lag_stat(K_work, lag_mat, "sd")

    lag_mean_values = np.array(list(lag_mean.values()), dtype=float)
    lag_median_values = np.array(list(lag_median.values()), dtype=float)
    lag_median_median = float(np.nanmedian(lag_median_values))
    lag_mean_sd = float(np.nanstd(lag_mean_values, ddof=1))
    lag_median_sd = float(np.nanstd(lag_median_values, ddof=1))

    def smooth_adjustment(lag_adjustment: dict[int, float]) -> dict[int, float]:
        lags = np.array(list(lag_adjustment.keys()), dtype=float)
        values = np.array(list(lag_adjustment.values()), dtype=float)
        valid = np.isfinite(lags) & np.isfinite(values)
        if valid.sum() < 4 or np.unique(lags[valid]).size < 4:
            return dict(lag_adjustment)

        # R uses smooth.spline(spar=s). This UnivariateSpline mapping is not
        # identical, but gives a smooth lag curve controlled by the same argument.
        spline = UnivariateSpline(lags[valid], values[valid], s=s * valid.sum())
        smoothed = spline(lags)
        return {int(lag): float(value) for lag, value in zip(lags, smoothed)}

    lag_adjust_raw_mean = lag_mean
    lag_adjust_raw_median = lag_median
    lag_adjust_raw_mean_smooth = smooth_adjustment(lag_adjust_raw_mean)
    lag_adjust_raw_median_smooth = smooth_adjustment(lag_adjust_raw_median)

    def build_adjustment_matrix(lag_adjustment: dict[int, float]) -> Array:
        adjust_mat = np.zeros((n, n), dtype=float)
        for lag, value in lag_adjustment.items():
            adjust_mat[lag_mat == lag] = value if np.isfinite(value) else 0.0
        adjust_mat[~np.isfinite(adjust_mat)] = 0.0
        return adjust_mat

    adjust_mat_raw_mean = build_adjustment_matrix(lag_adjust_raw_mean)
    adjust_mat_raw_median = build_adjustment_matrix(lag_adjust_raw_median)
    adjust_mat_raw_mean_smooth = build_adjustment_matrix(lag_adjust_raw_mean_smooth)
    adjust_mat_raw_median_smooth = build_adjustment_matrix(lag_adjust_raw_median_smooth)

    use_idx = lag_mat <= max_lag
    use_idx[lag_mat == 0] = False
    adjust_mat_raw_mean[~use_idx] = 0.0
    adjust_mat_raw_mean_smooth[~use_idx] = 0.0

    def residual_kernel(adjust_mat: Array) -> Array:
        K_resid = K - adjust_mat
        np.fill_diagonal(K_resid, 0.0)
        return (K_resid + K_resid.T) / 2.0

    def ratio_kernel(adjust_mat: Array, mm: float) -> Array:
        K_ratio = K.copy()
        ratio_idx = np.isfinite(adjust_mat) & (adjust_mat != 0)
        if np.isfinite(mm):
            K_ratio[ratio_idx] = K[ratio_idx] / adjust_mat[ratio_idx] * mm
        return (K_ratio + K_ratio.T) / 2.0

    return {
        "K_resid_raw_mean": residual_kernel(adjust_mat_raw_mean),
        "K_resid_raw_mean_smooth": residual_kernel(adjust_mat_raw_mean_smooth),
        "K_ratio_median": ratio_kernel(adjust_mat_raw_median, lag_median_median),
        "K_ratio_median_smooth": ratio_kernel(adjust_mat_raw_median_smooth, lag_median_median),
        "lag_mean": lag_mean,
        "lag_median": lag_median,
        "lag_sd": lag_sd,
        "lag_mean_sd": lag_mean_sd,
        "lag_median_sd": lag_median_sd,
        "lag_adjust_raw_mean_smooth": lag_adjust_raw_mean_smooth,
        "max_lag": max_lag,
    }


@jax.jit
def _lag_stats_jax(K_work: jax.Array, lag_mat: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    lags = jnp.arange(K_work.shape[0])

    def stats_for_lag(lag: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        values = jnp.where(lag_mat == lag, K_work, jnp.nan)
        mean = jnp.nanmean(values)
        median = jnp.nanmedian(values)
        sd = jnp.nanstd(values, ddof=1)
        return mean, median, sd

    lag_mean, lag_median, lag_sd = jax.vmap(stats_for_lag)(lags)
    return lag_mean, lag_median, lag_sd



@jax.jit
def _remove_lag_effect_jax_raw(K: jax.Array, max_lag: float) -> tuple[jax.Array, ...]:
    n = K.shape[0]
    diag_mask = jnp.eye(n, dtype=bool)
    K_work = jnp.where(diag_mask, jnp.nan, K)
    lag_mat = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])

    lag_mean, lag_median, lag_sd = _lag_stats_jax(K_work, lag_mat)
    lag_median_median = jnp.nanmedian(lag_median)
    lag_mean_sd = jnp.nanstd(lag_mean, ddof=1)
    lag_median_sd = jnp.nanstd(lag_median, ddof=1)

    adjust_mat_raw_mean = lag_mean[lag_mat]
    adjust_mat_raw_median = lag_median[lag_mat]
    adjust_mat_raw_mean = jnp.where(jnp.isfinite(adjust_mat_raw_mean), adjust_mat_raw_mean, 0.0)
    adjust_mat_raw_median = jnp.where(jnp.isfinite(adjust_mat_raw_median), adjust_mat_raw_median, 0.0)

    use_idx = (lag_mat <= max_lag) & (lag_mat != 0)
    adjust_mat_raw_mean = jnp.where(use_idx, adjust_mat_raw_mean, 0.0)

    K_resid_raw_mean = K - adjust_mat_raw_mean
    K_resid_raw_mean = jnp.where(diag_mask, 0.0, K_resid_raw_mean)
    K_resid_raw_mean = (K_resid_raw_mean + K_resid_raw_mean.T) / 2.0

    ratio_idx = jnp.isfinite(adjust_mat_raw_median) & (adjust_mat_raw_median != 0)
    K_ratio_median = jnp.where(
        ratio_idx & jnp.isfinite(lag_median_median),
        K / adjust_mat_raw_median * lag_median_median,
        K,
    )
    K_ratio_median = (K_ratio_median + K_ratio_median.T) / 2.0

    return (
        K_resid_raw_mean,
        K_ratio_median,
        lag_mean,
        lag_median,
        lag_sd,
        lag_mean_sd,
        lag_median_sd,
    )


@jax.jit
def _remove_lag_resid_raw_mean_jax(K: jax.Array, max_lag: float) -> jax.Array:
    n = K.shape[0]
    diag_mask = jnp.eye(n, dtype=bool)
    K_work = jnp.where(diag_mask, jnp.nan, K)
    lag_mat = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
    lag_mean, _, _ = _lag_stats_jax(K_work, lag_mat)

    adjust_mat = lag_mean[lag_mat]
    adjust_mat = jnp.where(jnp.isfinite(adjust_mat), adjust_mat, 0.0)
    use_idx = (lag_mat <= max_lag) & (lag_mat != 0)
    adjust_mat = jnp.where(use_idx, adjust_mat, 0.0)

    K_resid = K - adjust_mat
    K_resid = jnp.where(diag_mask, 0.0, K_resid)
    return (K_resid + K_resid.T) / 2.0


@jax.jit
def _remove_lag_ratio_median_jax(K: jax.Array, max_lag: float) -> jax.Array:
    del max_lag  # Kept for API symmetry; current R ratio path is not max-lag masked.
    n = K.shape[0]
    diag_mask = jnp.eye(n, dtype=bool)
    K_work = jnp.where(diag_mask, jnp.nan, K)
    lag_mat = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
    _, lag_median, _ = _lag_stats_jax(K_work, lag_mat)
    lag_median_median = jnp.nanmedian(lag_median)

    adjust_mat = lag_median[lag_mat]
    adjust_mat = jnp.where(jnp.isfinite(adjust_mat), adjust_mat, 0.0)
    ratio_idx = jnp.isfinite(adjust_mat) & (adjust_mat != 0)
    K_ratio = jnp.where(
        ratio_idx & jnp.isfinite(lag_median_median),
        K / adjust_mat * lag_median_median,
        K,
    )
    return (K_ratio + K_ratio.T) / 2.0


def _select_lag_adjusted_kernel_jax(K: jax.Array, adjust_type: str, max_lag: float) -> jax.Array:
    if adjust_type == "K_resid_raw_mean":
        return _remove_lag_resid_raw_mean_jax(K, max_lag)
    if adjust_type == "K_ratio_median":
        return _remove_lag_ratio_median_jax(K, max_lag)
    raise ValueError("adjust_type must be 'K_resid_raw_mean' or 'K_ratio_median'")


def remove_lag_effect_jax(K: Array, max_lag: float = np.inf) -> dict[str, Any]:
    """JAX-accelerated raw lag-effect removal.

    This accelerates the lag statistics, residual mean adjustment, and median
    ratio adjustment. Spline-smoothed adjustments are intentionally not included
    because the R/Python smoothing spline step is not JIT-compatible.
    """

    K_jax = jnp.asarray(K, dtype=jnp.float64)
    (
        K_resid_raw_mean,
        K_ratio_median,
        lag_mean,
        lag_median,
        lag_sd,
        lag_mean_sd,
        lag_median_sd
    ) = _remove_lag_effect_jax_raw(K_jax, float(max_lag))

    return {
        "K_resid_raw_mean": np.asarray(K_resid_raw_mean),
        "K_ratio_median": np.asarray(K_ratio_median),
        "lag_mean": np.asarray(lag_mean),
        "lag_median": np.asarray(lag_median),
        "lag_sd": np.asarray(lag_sd),
        "lag_mean_sd": float(lag_mean_sd),
        "lag_median_sd": float(lag_median_sd),
        "max_lag": max_lag,
    }


@jax.jit
def _kap_kernel_split_terms(K: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    n = K.shape[0]
    rows = jnp.arange(n)
    R_temp = jnp.sum(K, axis=1)

    Kx = jnp.zeros(n, dtype=K.dtype)
    Ky = jnp.zeros(n, dtype=K.dtype)
    Kx = Kx.at[1].set(jnp.sum(K[:2, :2]))
    Ky = Ky.at[1].set(jnp.sum(K[2:, 2:]))

    def step(carry: tuple[jax.Array, jax.Array], idx: jax.Array) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        prev_x, prev_y = carry
        temp_col = K[:, idx]
        add = jnp.sum(jnp.where(rows <= idx, temp_col, 0.0))
        subtract = R_temp[idx] - add
        next_x = prev_x + 2.0 * add
        next_y = prev_y - 2.0 * subtract
        return (next_x, next_y), (next_x, next_y)

    indices = jnp.arange(2, n - 2)
    _, values = lax.scan(step, (Kx[1], Ky[1]), indices)
    Kx = Kx.at[indices].set(values[0])
    Ky = Ky.at[indices].set(values[1])

    t = jnp.arange(1, n + 1, dtype=K.dtype)
    Kx = Kx / (t * (t - 1.0))
    Ky = Ky / ((n - t) * (n - t - 1.0))
    return Kx, Ky, R_temp


@jax.jit
def _safe_standardize(value: jax.Array, mean: jax.Array, variance: jax.Array) -> jax.Array:
    return (value - mean) / jnp.sqrt(jnp.where(variance > 0, variance, jnp.nan))


@jax.jit
def _kap_cpd_statistic_jax_core(
    K1: jax.Array,
    K2: jax.Array,
    r1: float,
    r2: float,
) -> tuple[jax.Array, ...]:
    n = K1.shape[0]
    dtype = K1.dtype
    t = jnp.arange(1, n + 1, dtype=dtype)
    n_float = jnp.asarray(n, dtype=dtype)

    K1x, K1y, R_temp1 = _kap_kernel_split_terms(K1)
    K2x, K2y, R_temp2 = _kap_kernel_split_terms(K2)

    R0_1 = jnp.sum(K1)
    R0_2 = jnp.sum(K2)
    mu_K1x = R0_1 / n_float / (n_float - 1.0)
    mu_K1y = mu_K1x
    mu_K2x = R0_2 / n_float / (n_float - 1.0)
    mu_K2y = mu_K2x

    A_1 = jnp.sum(K1**2)
    B_1 = jnp.sum(R_temp1**2) - A_1
    C_1 = R0_1**2 - 2.0 * A_1 - 4.0 * B_1

    A_2 = jnp.sum(K2**2)
    B_2 = jnp.sum(R_temp2**2) - A_2
    C_2 = R0_2**2 - 2.0 * A_2 - 4.0 * B_2

    A_cross = jnp.sum(K1 * K2)
    B_cross = jnp.sum(R_temp2 * R_temp1) - A_cross
    C_cross = R0_1 * R0_2 - 2.0 * A_cross - 4.0 * B_cross

    p1 = t * (t - 1.0) / n_float / (n_float - 1.0)
    p2 = p1 * (t - 2.0) / (n_float - 2.0)
    p3 = p2 * (t - 3.0) / (n_float - 3.0)

    nt = n_float - t
    q1 = nt * (nt - 1.0) / n_float / (n_float - 1.0)
    q2 = q1 * (nt - 2.0) / (n_float - 2.0)
    q3 = q2 * (nt - 3.0) / (n_float - 3.0)

    var_K1x = (2.0 * A_1 * p1 + 4.0 * B_1 * p2 + C_1 * p3) / t / t / (t - 1.0) / (t - 1.0) - mu_K1x**2
    var_K1y = (2.0 * A_1 * q1 + 4.0 * B_1 * q2 + C_1 * q3) / nt / nt / (nt - 1.0) / (nt - 1.0) - mu_K1y**2
    cov_K1x_K1y = C_1 / n_float / (n_float - 1.0) / (n_float - 2.0) / (n_float - 3.0) - mu_K1x * mu_K1y

    var_K2x = (2.0 * A_2 * p1 + 4.0 * B_2 * p2 + C_2 * p3) / t / t / (t - 1.0) / (t - 1.0) - mu_K2x**2
    var_K2y = (2.0 * A_2 * q1 + 4.0 * B_2 * q2 + C_2 * q3) / nt / nt / (nt - 1.0) / (nt - 1.0) - mu_K2y**2
    cov_K2x_K2y = C_2 / n_float / (n_float - 1.0) / (n_float - 2.0) / (n_float - 3.0) - mu_K2x * mu_K2y

    cov_K1x_K2x = (2.0 * A_cross * p1 + 4.0 * B_cross * p2 + C_cross * p3) / t / t / (t - 1.0) / (t - 1.0) - mu_K2x * mu_K1x
    cov_K1y_K2y = (2.0 * A_cross * q1 + 4.0 * B_cross * q2 + C_cross * q3) / nt / nt / (nt - 1.0) / (nt - 1.0) - mu_K1y * mu_K2y
    cov_K1x_K2y = C_cross / n_float / (n_float - 1.0) / (n_float - 2.0) / (n_float - 3.0) - mu_K1x * mu_K2y

    D_weight1 = t * (t - 1.0) / (n_float * (n_float - 1.0))
    D_weight2 = -nt * (nt - 1.0) / (n_float * (n_float - 1.0))
    w_weight1 = t / n_float
    w_weight2 = nt / n_float

    mean_D1 = mu_K1x * D_weight1 + mu_K1y * D_weight2
    var_D1 = D_weight1**2 * var_K1x + D_weight2**2 * var_K1y + 2.0 * D_weight1 * D_weight2 * cov_K1x_K1y
    D1 = K1x * D_weight1 + K1y * D_weight2

    mean_W1 = mu_K1x * w_weight1 + mu_K1y * w_weight2
    var_W1 = var_K1x * w_weight1**2 + var_K1y * w_weight2**2 + 2.0 * w_weight1 * w_weight2 * cov_K1x_K1y
    W1 = K1x * w_weight1 + K1y * w_weight2

    mean_W1_r1 = r1 * mu_K1x * w_weight1 + mu_K1y * w_weight2
    var_W1_r1 = var_K1x * w_weight1**2 * r1**2 + var_K1y * w_weight2**2 + 2.0 * r1 * w_weight1 * w_weight2 * cov_K1x_K1y
    W1_r1 = K1x * w_weight1 * r1 + K1y * w_weight2

    mean_W1_r2 = r2 * mu_K1x * w_weight1 + mu_K1y * w_weight2
    var_W1_r2 = var_K1x * w_weight1**2 * r2**2 + var_K1y * w_weight2**2 + 2.0 * r2 * w_weight1 * w_weight2 * cov_K1x_K1y
    W1_r2 = K1x * w_weight1 * r2 + K1y * w_weight2

    mean_D2 = mu_K2x * D_weight1 + mu_K2y * D_weight2
    var_D2 = D_weight1**2 * var_K2x + D_weight2**2 * var_K2y + 2.0 * D_weight1 * D_weight2 * cov_K2x_K2y
    D2 = K2x * D_weight1 + K2y * D_weight2

    mean_W2 = mu_K2x * w_weight1 + mu_K2y * w_weight2
    var_W2 = var_K2x * w_weight1**2 + var_K2y * w_weight2**2 + 2.0 * w_weight1 * w_weight2 * cov_K2x_K2y
    W2 = K2x * w_weight1 + K2y * w_weight2

    mean_W2_r1 = r1 * mu_K2x * w_weight1 + mu_K2y * w_weight2
    var_W2_r1 = var_K2x * w_weight1**2 * r1**2 + var_K2y * w_weight2**2 + 2.0 * r1 * w_weight1 * w_weight2 * cov_K2x_K2y
    W2_r1 = r1 * K2x * w_weight1 + K2y * w_weight2

    mean_W2_r2 = mu_K2x * w_weight1 * r2 + mu_K2y * w_weight2
    var_W2_r2 = var_K2x * w_weight1**2 * r2**2 + var_K2y * w_weight2**2 + 2.0 * r2 * w_weight1 * w_weight2 * cov_K2x_K2y
    W2_r2 = K2x * w_weight1 * r2 + K2y * w_weight2

    cov_w1_w2 = w_weight1**2 * cov_K1x_K2x + w_weight2**2 * cov_K1y_K2y + 2.0 * w_weight1 * w_weight2 * cov_K1x_K2y
    cov_D1_D2 = D_weight1**2 * cov_K1x_K2x + D_weight2**2 * cov_K1y_K2y + 2.0 * D_weight1 * D_weight2 * cov_K1x_K2y
    #cov_w1_w2_r1 = r1**2 * w_weight1**2 * cov_K1x_K2x + w_weight2**2 * cov_K1y_K2y + 2.0 * r1 * w_weight1 * w_weight2 * cov_K1x_K2y
    #cov_w1_w2_r2 = r2**2 * w_weight1**2 * cov_K1x_K2x + w_weight2**2 * cov_K1y_K2y + 2.0 * r2 * w_weight1 * w_weight2 * cov_K1x_K2y

    t2 = (A_2 + B_2) - (2.0 * A_2 + 4.0 * B_2 + C_2) / n_float
    t1 = (A_1 + B_1) - (2.0 * A_1 + 4.0 * B_1 + C_1) / n_float
    t_cross = (A_cross + B_cross) - (2.0 * A_cross + 4.0 * B_cross + C_cross) / n_float
    x = (t2 - t_cross) / (t1 - t_cross)

    m1 = (n_float - 2.0) * A_1 + (2.0 * A_1 + 4.0 * B_1 + C_1) / (n_float - 1.0) - 2.0 * (A_1 + B_1)
    m2 = (n_float - 2.0) * A_2 + (2.0 * A_2 + 4.0 * B_2 + C_2) / (n_float - 1.0) - 2.0 * (A_2 + B_2)
    m_cross = (n_float - 2.0) * A_cross + (2.0 * A_cross + 4.0 * B_cross + C_cross) / (n_float - 1.0) - 2.0 * (A_cross + B_cross)
    a = (m2 - m_cross) / (m1 - m_cross)

    ZW1_r1 = _safe_standardize(W1_r1, mean_W1_r1, var_W1_r1)
    ZW1_r2 = _safe_standardize(W1_r2, mean_W1_r2, var_W1_r2)
    ZW1 = _safe_standardize(W1, mean_W1, var_W1)
    ZD1 = _safe_standardize(D1, mean_D1, var_D1)
    ZW2_r1 = _safe_standardize(W2_r1, mean_W2_r1, var_W2_r1)
    ZW2_r2 = _safe_standardize(W2_r2, mean_W2_r2, var_W2_r2)
    ZW2 = _safe_standardize(W2, mean_W2, var_W2)
    ZD2 = _safe_standardize(D2, mean_D2, var_D2)

    Diff_W = (W1 - W2 - mean_W1 + mean_W2) / jnp.sqrt(var_W1 + var_W2 - 2.0 * cov_w1_w2)
    Diff_D = (D1 - D2 - mean_D1 + mean_D2) / jnp.sqrt(var_D1 + var_D2 - 2.0 * cov_D1_D2)
    Sum_W = (a * W1 + W2 - a * mean_W1 - mean_W2) / jnp.sqrt(a**2 * var_W1 + var_W2 + 2.0 * a * cov_w1_w2)
    Sum_D = (x * D1 + D2 - x * mean_D1 - mean_D2) / jnp.sqrt(x**2 * var_D1 + var_D2 + 2.0 * x * cov_D1_D2)
    #Sum_W_r1 = (W1_r1 + W2_r1 - mean_W1_r1 - mean_W2_r1) / jnp.sqrt(var_W1_r1 + var_W2_r1 + 2.0 * cov_w1_w2_r1)
    #Sum_W_r2 = (W1_r2 + W2_r2 - mean_W1_r2 - mean_W2_r2) / jnp.sqrt(var_W1_r2 + var_W2_r2 + 2.0 * cov_w1_w2_r2)
    #Diff_W_r1 = (W1_r1 - W2_r1 - mean_W1_r1 + mean_W2_r1) / jnp.sqrt(var_W1_r1 + var_W2_r1 - 2.0 * cov_w1_w2_r1)
    #Diff_W_r2 = (W1_r2 - W2_r2 - mean_W1_r2 + mean_W2_r2) / jnp.sqrt(var_W1_r2 + var_W2_r2 - 2.0 * cov_w1_w2_r2)
    S = Sum_W**2 + Sum_D**2 + Diff_W**2 + Diff_D**2

    return (
        ZW1_r1, ZW1_r2, ZW1, ZD1,
        ZW2_r1, ZW2_r2, ZW2, ZD2,
        #Sum_W_r1, Diff_W_r1, Sum_W_r2, Diff_W_r2,
        Sum_D, Diff_D, Diff_W, Sum_W, S,
        a, x,
    )


@partial(jax.jit, static_argnames=("n0_idx", "n1_idx_exclusive", "B"))
def _kap_cpd_permutation_pvalue_jax_core(
    key: jax.Array,
    K1: jax.Array,
    K2: jax.Array,
    r1: float,
    r2: float,
    n0_idx: int,
    n1_idx_exclusive: int,
    B: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    observed = _kap_cpd_statistic_jax_core(K1, K2, r1, r2)
    observed_S = observed[12]
    observed_max = jnp.nanmax(observed_S[n0_idx:n1_idx_exclusive])

    keys = jrandom.split(key, B)
    n = K1.shape[0]

    def permutation_max_S(subkey: jax.Array) -> jax.Array:
        perm = jrandom.permutation(subkey, n)
        K1_perm = K1[perm[:, None], perm[None, :]]
        K2_perm = K2[perm[:, None], perm[None, :]]
        permuted = _kap_cpd_statistic_jax_core(K1_perm, K2_perm, r1, r2)
        S_perm = permuted[12]
        return jnp.nanmax(S_perm[n0_idx:n1_idx_exclusive])

    max_S = jax.vmap(permutation_max_S)(keys)
    pvalue = jnp.minimum(1.0, jnp.mean(max_S >= observed_max))
    return pvalue, observed_max, max_S


def kap_cpd_permutation_pvalue(
    K1: Array,
    K2: Array,
    B: int = 1000,
    r1: float = 0.5,
    r2: float = 2.0,
    n: int | None = None,
    n0: int | None = None,
    n1: int | None = None,
    seed: int = 0,
    return_distribution: bool = False,
) -> float | dict[str, Any]:
    """Permutation p-value for the KAP-CPD S statistic, accelerated with JAX.

    This mirrors the R `permpval2` function: it permutes rows/columns of `K1`
    and `K2` together, recomputes the S scan statistic, and compares the maximum
    permuted S over `[n0, n1]` to the observed maximum S. The scan window uses
    the same 1-based indexing convention as the R code.
    """

    if B < 1:
        raise ValueError("B must be at least 1")

    K1_np = np.asarray(K1, dtype=np.float64)
    K2_np = np.asarray(K2, dtype=np.float64)
    if K1_np.shape != K2_np.shape or K1_np.ndim != 2 or K1_np.shape[0] != K1_np.shape[1]:
        raise ValueError("K1 and K2 must be square matrices with the same shape")

    n_actual = K1_np.shape[0]
    if n is None:
        n = n_actual
    elif n != n_actual:
        raise ValueError(f"n={n} does not match matrix size {n_actual}")

    if n0 is None:
        n0 = int(np.ceil(0.05 * n))
    if n1 is None:
        n1 = int(np.floor(0.95 * n))
    n0 = max(n0, 2)
    n1 = min(n1, n - 2)
    if n0 > n1:
        raise ValueError("n0 must be <= n1 after boundary adjustment")

    pvalue, observed_max, max_S = _kap_cpd_permutation_pvalue_jax_core(
        jrandom.PRNGKey(seed),
        jnp.asarray(K1_np),
        jnp.asarray(K2_np),
        float(r1),
        float(r2),
        n0 - 1,
        n1,
        int(B),
    )

    if return_distribution:
        return {
            "pvalue": float(pvalue),
            "observed_max_S": float(observed_max),
            "permuted_max_S": np.asarray(max_S),
            "B": B,
            "n0": n0,
            "n1": n1,
            "seed": seed,
        }
    return float(pvalue)


def kap_cpd_permutation_pvalue_batched(
    K1: Array,
    K2: Array,
    B: int = 1000,
    batch_size: int = 50,
    r1: float = 0.5,
    r2: float = 2.0,
    n: int | None = None,
    n0: int | None = None,
    n1: int | None = None,
    seed: int = 0,
    return_distribution: bool = False,
) -> float | dict[str, Any]:
    """Memory-batched permutation p-value for the original KAP-CPD S statistic.

    This computes the same quantity as `kap_cpd_permutation_pvalue`, but splits
    permutations into batches so JAX does not materialize all B permutations at
    once. This is safer for Colab/GPU memory.
    """

    if B < 1:
        raise ValueError("B must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    K1_np = np.asarray(K1, dtype=np.float64)
    K2_np = np.asarray(K2, dtype=np.float64)
    if K1_np.shape != K2_np.shape or K1_np.ndim != 2 or K1_np.shape[0] != K1_np.shape[1]:
        raise ValueError("K1 and K2 must be square matrices with the same shape")

    n_actual = K1_np.shape[0]
    if n is None:
        n = n_actual
    elif n != n_actual:
        raise ValueError(f"n={n} does not match matrix size {n_actual}")

    if n0 is None:
        n0 = int(np.ceil(0.05 * n))
    if n1 is None:
        n1 = int(np.floor(0.95 * n))
    n0 = max(n0, 2)
    n1 = min(n1, n - 2)
    if n0 > n1:
        raise ValueError("n0 must be <= n1 after boundary adjustment")

    K1_jax = jnp.asarray(K1_np)
    K2_jax = jnp.asarray(K2_np)
    key = jrandom.PRNGKey(seed)
    remaining = int(B)
    exceed_count = 0
    observed_max_value: float | None = None
    max_values: list[np.ndarray] = []

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        key, subkey = jrandom.split(key)
        _, observed_max, max_S = _kap_cpd_permutation_pvalue_jax_core(
            subkey,
            K1_jax,
            K2_jax,
            float(r1),
            float(r2),
            n0 - 1,
            n1,
            current_batch,
        )
        observed_max_float = float(observed_max)
        if observed_max_value is None:
            observed_max_value = observed_max_float
        max_S_np = np.asarray(max_S)
        exceed_count += int(np.sum(max_S_np >= observed_max_float))
        if return_distribution:
            max_values.append(max_S_np)
        remaining -= current_batch

    pvalue = min(1.0, exceed_count / B)
    if return_distribution:
        return {
            "pvalue": pvalue,
            "observed_max_S": observed_max_value,
            "permuted_max_S": np.concatenate(max_values) if max_values else np.array([]),
            "B": B,
            "batch_size": batch_size,
            "n0": n0,
            "n1": n1,
            "seed": seed,
        }
    return pvalue


def kap_cpd_permutation_pvalue_dependent_batched(
    K1: Array,
    K2: Array,
    adjust_type: str,
    max_lag: float,
    B: int = 1000,
    batch_size: int = 50,
    r1: float = 0.5,
    r2: float = 2.0,
    n: int | None = None,
    n0: int | None = None,
    n1: int | None = None,
    seed: int = 0,
    return_distribution: bool = False,
) -> float | dict[str, Any]:
    """Memory-batched permutation p-value for lag-adjusted KAP-CPD.

    Supports `adjust_type="K_resid_raw_mean"` and `adjust_type="K_ratio_median"`.
    """

    if B < 1:
        raise ValueError("B must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    K1_np = np.asarray(K1, dtype=np.float64)
    K2_np = np.asarray(K2, dtype=np.float64)
    if K1_np.shape != K2_np.shape or K1_np.ndim != 2 or K1_np.shape[0] != K1_np.shape[1]:
        raise ValueError("K1 and K2 must be square matrices with the same shape")

    n_actual = K1_np.shape[0]
    if n is None:
        n = n_actual
    elif n != n_actual:
        raise ValueError(f"n={n} does not match matrix size {n_actual}")

    if n0 is None:
        n0 = int(np.ceil(0.05 * n))
    if n1 is None:
        n1 = int(np.floor(0.95 * n))
    n0 = max(n0, 2)
    n1 = min(n1, n - 2)
    if n0 > n1:
        raise ValueError("n0 must be <= n1 after boundary adjustment")

    K1_jax = jnp.asarray(K1_np)
    K2_jax = jnp.asarray(K2_np)
    key = jrandom.PRNGKey(seed)
    remaining = int(B)
    exceed_count = 0
    observed_max_value: float | None = None
    max_values: list[np.ndarray] = []

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        key, subkey = jrandom.split(key)
        _, observed_max, max_S = _kap_cpd_permutation_pvalue_dependent_jax_core(
            subkey,
            K1_jax,
            K2_jax,
            float(max_lag),
            adjust_type,
            float(r1),
            float(r2),
            n0 - 1,
            n1,
            current_batch,
        )
        observed_max_float = float(observed_max)
        if observed_max_value is None:
            observed_max_value = observed_max_float
        max_S_np = np.asarray(max_S)
        exceed_count += int(np.sum(max_S_np >= observed_max_float))
        if return_distribution:
            max_values.append(max_S_np)
        remaining -= current_batch

    pvalue = min(1.0, exceed_count / B)
    if return_distribution:
        return {
            "pvalue": pvalue,
            "observed_max_S": observed_max_value,
            "permuted_max_S": np.concatenate(max_values) if max_values else np.array([]),
            "B": B,
            "batch_size": batch_size,
            "n0": n0,
            "n1": n1,
            "seed": seed,
        }
    return pvalue


def _scan_summary(scan: Array, n0: int, n1: int, use_abs: bool = False, field: str = "Zmax") -> dict[str, Any]:
    window = np.abs(scan[n0 - 1:n1]) if use_abs else scan[n0 - 1:n1]
    best = int(np.nanargmax(window))
    return {
        "scan": scan,
        field: float(window[best]),
        "tauhat": n0 + best,
    }


def kap_cpd_statistic(
    K1: Array,
    K2: Array,
    r1: float = 0.5,
    r2: float = 2.0,
    n: int | None = None,
    n0: int | None = None,
    n1: int | None = None,
) -> dict[str, Any]:
    """JAX-accelerated Python version of the R `KAP_CPD_statistic` function.

    The returned dictionary mirrors the R list names. Arrays are length `n`, and
    `tauhat` values use the same 1-based time indexing as the original R code.
    """

    K1_np = np.asarray(K1, dtype=np.float64)
    K2_np = np.asarray(K2, dtype=np.float64)
    if K1_np.shape != K2_np.shape or K1_np.ndim != 2 or K1_np.shape[0] != K1_np.shape[1]:
        raise ValueError("K1 and K2 must be square matrices with the same shape")

    n_actual = K1_np.shape[0]
    if n is None:
        n = n_actual
    elif n != n_actual:
        raise ValueError(f"n={n} does not match matrix size {n_actual}")

    if n0 is None:
        n0 = int(np.ceil(0.05 * n))
    if n1 is None:
        n1 = int(np.floor(0.95 * n))
    n0 = max(n0, 2)
    n1 = min(n1, n - 2)
    if n0 > n1:
        raise ValueError("n0 must be <= n1 after boundary adjustment")

    outputs = _kap_cpd_statistic_jax_core(
        jnp.asarray(K1_np),
        jnp.asarray(K2_np),
        float(r1),
        float(r2),
    )
    arrays = [np.asarray(item) if getattr(item, "shape", ()) else float(item) for item in outputs]
    (
        ZW1_r1, ZW1_r2, ZW1, ZD1,
        ZW2_r1, ZW2_r2, ZW2, ZD2,
        #Sum_W_r1, Diff_W_r1, Sum_W_r2, Diff_W_r2,
        Sum_D, Diff_D, Diff_W, Sum_W, S,
        a, x,
    ) = arrays

    return {
        "ZW1_r1": _scan_summary(ZW1_r1, n0, n1),
        "ZW1_r2": _scan_summary(ZW1_r2, n0, n1),
        "ZW1": _scan_summary(ZW1, n0, n1),
        "ZD1": _scan_summary(ZD1, n0, n1, use_abs=True),
        "ZW2_r1": _scan_summary(ZW2_r1, n0, n1),
        "ZW2_r2": _scan_summary(ZW2_r2, n0, n1),
        "ZW2": _scan_summary(ZW2, n0, n1),
        "ZD2": _scan_summary(ZD2, n0, n1, use_abs=True),
        "weightW": float(a),
        "weightD": float(x),
        #"W_sum_r1": _scan_summary(Sum_W_r1, n0, n1, field="max"),
        #"W_diff_r1": _scan_summary(Diff_W_r1, n0, n1, use_abs=True, field="max"),
        #"W_sum_r2": _scan_summary(Sum_W_r2, n0, n1, field="max"),
        #"W_diff_r2": _scan_summary(Diff_W_r2, n0, n1, use_abs=True, field="max"),
        "D_sum": _scan_summary(Sum_D, n0, n1, use_abs=True, field="max"),
        "D_diff": _scan_summary(Diff_D, n0, n1, use_abs=True, field="max"),
        "W_diff": _scan_summary(Diff_W, n0, n1, use_abs=True, field="max"),
        "W_sum": _scan_summary(Sum_W, n0, n1, field="max"),
        "S": _scan_summary(S, n0, n1, field="max"),
    }



def kap_cpd_statistic_dependent(
    K1_raw: Array,
    K2_raw: Array,
    adjust_type:str,
    max_lag:float,
    r1: float = 0.5,
    r2: float = 2.0,
    n: int | None = None,
    n0: int | None = None,
    n1: int | None = None,
) -> dict[str, Any]:
    """JAX-accelerated Python version of the R `KAP_CPD_statistic` function.

    The returned dictionary mirrors the R list names. Arrays are length `n`, and
    `tauhat` values use the same 1-based time indexing as the original R code.
    """
    K1 = np.asarray(_select_lag_adjusted_kernel_jax(jnp.asarray(K1_raw, dtype=jnp.float64), adjust_type, float(max_lag)))
    K2 = np.asarray(_select_lag_adjusted_kernel_jax(jnp.asarray(K2_raw, dtype=jnp.float64), adjust_type, float(max_lag)))
    K1_np = np.asarray(K1, dtype=np.float64)
    K2_np = np.asarray(K2, dtype=np.float64)
    if K1_np.shape != K2_np.shape or K1_np.ndim != 2 or K1_np.shape[0] != K1_np.shape[1]:
        raise ValueError("K1 and K2 must be square matrices with the same shape")

    n_actual = K1_np.shape[0]
    if n is None:
        n = n_actual
    elif n != n_actual:
        raise ValueError(f"n={n} does not match matrix size {n_actual}")

    if n0 is None:
        n0 = int(np.ceil(0.05 * n))
    if n1 is None:
        n1 = int(np.floor(0.95 * n))
    n0 = max(n0, 2)
    n1 = min(n1, n - 2)
    if n0 > n1:
        raise ValueError("n0 must be <= n1 after boundary adjustment")

    outputs = _kap_cpd_statistic_jax_core(
        jnp.asarray(K1_np),
        jnp.asarray(K2_np),
        float(r1),
        float(r2),
    )
    arrays = [np.asarray(item) if getattr(item, "shape", ()) else float(item) for item in outputs]
    (
        ZW1_r1, ZW1_r2, ZW1, ZD1,
        ZW2_r1, ZW2_r2, ZW2, ZD2,
        #Sum_W_r1, Diff_W_r1, Sum_W_r2, Diff_W_r2,
        Sum_D, Diff_D, Diff_W, Sum_W, S,
        a, x,
    ) = arrays

    return {
        "ZW1_r1": _scan_summary(ZW1_r1, n0, n1),
        "ZW1_r2": _scan_summary(ZW1_r2, n0, n1),
        "ZW1": _scan_summary(ZW1, n0, n1),
        "ZD1": _scan_summary(ZD1, n0, n1, use_abs=True),
        "ZW2_r1": _scan_summary(ZW2_r1, n0, n1),
        "ZW2_r2": _scan_summary(ZW2_r2, n0, n1),
        "ZW2": _scan_summary(ZW2, n0, n1),
        "ZD2": _scan_summary(ZD2, n0, n1, use_abs=True),
        "weightW": float(a),
        "weightD": float(x),
        #"W_sum_r1": _scan_summary(Sum_W_r1, n0, n1, field="max"),
        #"W_diff_r1": _scan_summary(Diff_W_r1, n0, n1, use_abs=True, field="max"),
        #"W_sum_r2": _scan_summary(Sum_W_r2, n0, n1, field="max"),
        #"W_diff_r2": _scan_summary(Diff_W_r2, n0, n1, use_abs=True, field="max"),
        "D_sum": _scan_summary(Sum_D, n0, n1, use_abs=True, field="max"),
        "D_diff": _scan_summary(Diff_D, n0, n1, use_abs=True, field="max"),
        "W_diff": _scan_summary(Diff_W, n0, n1, use_abs=True, field="max"),
        "W_sum": _scan_summary(Sum_W, n0, n1, field="max"),
        "S": _scan_summary(S, n0, n1, field="max"),
    }

@partial(jax.jit, static_argnames=("adjust_type", "n0_idx", "n1_idx_exclusive", "B"))
def _kap_cpd_permutation_pvalue_dependent_jax_core(
    key: jax.Array,
    K1_raw: jax.Array,
    K2_raw: jax.Array,
    max_lag: float,
    adjust_type: str,
    r1: float,
    r2: float,
    n0_idx: int,
    n1_idx_exclusive: int,
    B: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    def adjust_kernel(K_raw: jax.Array) -> jax.Array:
        return _select_lag_adjusted_kernel_jax(K_raw, adjust_type, max_lag)

    K1 = adjust_kernel(K1_raw)
    K2 = adjust_kernel(K2_raw)
    observed = _kap_cpd_statistic_jax_core(K1, K2, r1, r2)
    observed_S = observed[12]
    observed_max = jnp.nanmax(observed_S[n0_idx:n1_idx_exclusive])

    keys = jrandom.split(key, B)
    n = K1.shape[0]

    def permutation_max_S(subkey: jax.Array) -> jax.Array:
        perm = jrandom.permutation(subkey, n)
        K1_raw_perm = K1_raw[perm[:, None], perm[None, :]]
        K2_raw_perm = K2_raw[perm[:, None], perm[None, :]]
        K1_perm = adjust_kernel(K1_raw_perm)
        K2_perm = adjust_kernel(K2_raw_perm)
        permuted = _kap_cpd_statistic_jax_core(K1_perm, K2_perm, r1, r2)
        S_perm = permuted[12]
        return jnp.nanmax(S_perm[n0_idx:n1_idx_exclusive])

    max_S = jax.vmap(permutation_max_S)(keys)
    pvalue = jnp.minimum(1.0, jnp.mean(max_S >= observed_max))
    return pvalue, observed_max, max_S


def kap_cpd_permutation_pvalue_dependent(
    K1: Array,
    K2: Array,
    adjust_type: str,
    max_lag: float,
    B: int = 1000,
    r1: float = 0.5,
    r2: float = 2.0,
    n: int | None = None,
    n0: int | None = None,
    n1: int | None = None,
    seed: int = 0,
    return_distribution: bool = False,
) -> float | dict[str, Any]:
    """Lag-adjusted permutation p-value for the KAP-CPD S statistic."""

    if B < 1:
        raise ValueError("B must be at least 1")

    K1_np = np.asarray(K1, dtype=np.float64)
    K2_np = np.asarray(K2, dtype=np.float64)
    if K1_np.shape != K2_np.shape or K1_np.ndim != 2 or K1_np.shape[0] != K1_np.shape[1]:
        raise ValueError("K1 and K2 must be square matrices with the same shape")

    n_actual = K1_np.shape[0]
    if n is None:
        n = n_actual
    elif n != n_actual:
        raise ValueError(f"n={n} does not match matrix size {n_actual}")

    if n0 is None:
        n0 = int(np.ceil(0.05 * n))
    if n1 is None:
        n1 = int(np.floor(0.95 * n))
    n0 = max(n0, 2)
    n1 = min(n1, n - 2)
    if n0 > n1:
        raise ValueError("n0 must be <= n1 after boundary adjustment")

    pvalue, observed_max, max_S = _kap_cpd_permutation_pvalue_dependent_jax_core(
        jrandom.PRNGKey(seed),
        jnp.asarray(K1_np),
        jnp.asarray(K2_np),
        float(max_lag),
        adjust_type,
        float(r1),
        float(r2),
        n0 - 1,
        n1,
        int(B),
    )

    if return_distribution:
        return {
            "pvalue": float(pvalue),
            "observed_max_S": float(observed_max),
            "permuted_max_S": np.asarray(max_S),
            "B": B,
            "n0": n0,
            "n1": n1,
            "seed": seed,
        }
    return float(pvalue)


def plot_matrix_heatmap(M: Array, title: str = "Kernel Matrix Heat Map", ax: Any | None = None) -> Any:
    """Plot a matrix heat map like the ggplot helper in the R file."""

    M = np.asarray(M, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(M, ax=ax, cmap="viridis", cbar=True, square=True)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    return ax
