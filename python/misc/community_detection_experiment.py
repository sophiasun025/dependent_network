import numpy as np
from typing import Callable, Dict, Tuple, Optional


def compute_Zw(A: np.ndarray, x: np.ndarray) -> float:
    """
    Placeholder for Z_w(x).

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix, shape (n, n).
    x : np.ndarray
        Binary label vector, shape (n,). Entries should be 0 or 1.

    Returns
    -------
    float
        The standardized weighted within-edge statistic Z_w(x).
    """
    raise NotImplementedError("Fill in the formula for Z_w(x).")


def compute_Zd(A: np.ndarray, x: np.ndarray) -> float:
    """
    Placeholder for Z_d(x).

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix, shape (n, n).
    x : np.ndarray
        Binary label vector, shape (n,). Entries should be 0 or 1.

    Returns
    -------
    float
        The standardized difference statistic Z_d(x).
    """
    raise NotImplementedError("Fill in the formula for Z_d(x).")


def random_binary_labels(
    n: int,
    min_group_size: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a random non-degenerate binary partition.
    """
    if rng is None:
        rng = np.random.default_rng()

    while True:
        x = rng.integers(0, 2, size=n)
        n1 = np.sum(x == 1)
        n0 = n - n1

        if n0 >= min_group_size and n1 >= min_group_size:
            return x


def flip_label(x: np.ndarray, i: int) -> np.ndarray:
    """
    Return a copy of x with node i flipped.
    """
    x_new = x.copy()
    x_new[i] = 1 - x_new[i]
    return x_new


def greedy_local_search(
    A: np.ndarray,
    score_func: Callable[[np.ndarray, np.ndarray], float],
    x_init: np.ndarray,
    maximize: bool = True,
    min_group_size: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """
    Greedy single-node-flip local search.

    At each step, try flipping every node and accept the flip that gives
    the largest improvement in the objective.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix.
    score_func : callable
        Function score_func(A, x), for example compute_Zw or compute_Zd.
    x_init : np.ndarray
        Initial binary label vector.
    maximize : bool
        If True, maximize the score. If False, minimize the score.
    min_group_size : int
        Minimum allowed size for each group.
    max_iter : int
        Maximum number of greedy updates.
    tol : float
        Minimum improvement threshold.

    Returns
    -------
    x_best : np.ndarray
        Locally optimal binary label vector.
    score_best : float
        Score of the locally optimal label vector.
    """
    n = A.shape[0]
    x = x_init.copy()

    current_score = score_func(A, x)

    for _ in range(max_iter):
        best_i = None
        best_score = current_score

        for i in range(n):
            x_candidate = flip_label(x, i)

            n1 = np.sum(x_candidate == 1)
            n0 = n - n1

            if n0 < min_group_size or n1 < min_group_size:
                continue

            candidate_score = score_func(A, x_candidate)

            if maximize:
                improved = candidate_score > best_score + tol
            else:
                improved = candidate_score < best_score - tol

            if improved:
                best_score = candidate_score
                best_i = i

        if best_i is None:
            break

        x[best_i] = 1 - x[best_i]
        current_score = best_score

    return x, current_score


def multi_start_fit(
    A: np.ndarray,
    score_func: Callable[[np.ndarray, np.ndarray], float],
    maximize: bool = True,
    n_starts: int = 20,
    min_group_size: int = 1,
    max_iter: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """
    Run greedy local search from many random initializations.

    This is useful because the objective is nonconvex and greedy search
    can get stuck in local optima.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]

    best_x = None
    best_score = -np.inf if maximize else np.inf

    for _ in range(n_starts):
        x_init = random_binary_labels(
            n=n,
            min_group_size=min_group_size,
            rng=rng,
        )

        x_hat, score_hat = greedy_local_search(
            A=A,
            score_func=score_func,
            x_init=x_init,
            maximize=maximize,
            min_group_size=min_group_size,
            max_iter=max_iter,
        )

        if maximize:
            is_better = score_hat > best_score
        else:
            is_better = score_hat < best_score

        if is_better:
            best_x = x_hat
            best_score = score_hat

    return best_x, best_score


def penalized_likelihood_placeholder(A: np.ndarray, x: np.ndarray) -> float:
    """
    Placeholder for the UBSea selection criterion.

    In the paper, after fitting candidate labels using Z_w^max, Z_w^min,
    and Z_d, they choose among the candidate partitions using a criterion
    such as penalized likelihood.

    You can replace this function with your implementation.

    Larger value is assumed to be better here.
    """
    raise NotImplementedError("Fill in the penalized likelihood criterion.")


def simple_selection_criterion(A: np.ndarray, x: np.ndarray) -> float:
    """
    A simple placeholder selection criterion.

    This is NOT the exact paper criterion.
    It just computes within-group edge density contrast as a toy example.

    Replace this with the paper's penalized likelihood if needed.
    """
    group1 = np.where(x == 1)[0]
    group0 = np.where(x == 0)[0]

    if len(group1) <= 1 or len(group0) <= 1:
        return -np.inf

    A11 = A[np.ix_(group1, group1)]
    A00 = A[np.ix_(group0, group0)]
    A10 = A[np.ix_(group1, group0)]

    within_edges = A11.sum() / 2 + A00.sum() / 2
    between_edges = A10.sum()

    return within_edges - between_edges


def ubsea_fit(
    A: np.ndarray,
    n_starts: int = 20,
    min_group_size: int = 2,
    max_iter: int = 1000,
    rng: Optional[np.random.Generator] = None,
    selection_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Dict[str, object]:
    """
    UBSea-style fitting strategy.

    Fits three candidate partitions:

    1. Z_w max: associative structure.
    2. Z_w min: disassociative structure.
    3. Z_d max: core-periphery structure.

    Then chooses among the three using a selection criterion.

    Parameters
    ----------
    A : np.ndarray
        Undirected adjacency matrix, shape (n, n).
    n_starts : int
        Number of random starts for each objective.
    min_group_size : int
        Minimum allowed community size.
    max_iter : int
        Maximum local-search iterations.
    rng : np.random.Generator
        Random generator.
    selection_func : callable
        Function selection_func(A, x). Larger is better.
        If None, uses a toy placeholder criterion.

    Returns
    -------
    result : dict
        Contains candidate labels, candidate scores, selected label,
        and selected model type.
    """
    if rng is None:
        rng = np.random.default_rng()

    if selection_func is None:
        selection_func = simple_selection_criterion

    # Candidate 1: associative structure
    x_zw_max, score_zw_max = multi_start_fit(
        A=A,
        score_func=compute_Zw,
        maximize=True,
        n_starts=n_starts,
        min_group_size=min_group_size,
        max_iter=max_iter,
        rng=rng,
    )

    # Candidate 2: disassociative structure
    x_zw_min, score_zw_min = multi_start_fit(
        A=A,
        score_func=compute_Zw,
        maximize=False,
        n_starts=n_starts,
        min_group_size=min_group_size,
        max_iter=max_iter,
        rng=rng,
    )

    # Candidate 3: core-periphery structure
    x_zd, score_zd = multi_start_fit(
        A=A,
        score_func=compute_Zd,
        maximize=True,
        n_starts=n_starts,
        min_group_size=min_group_size,
        max_iter=max_iter,
        rng=rng,
    )

    candidates = {
        "associative_Zw_max": {
            "labels": x_zw_max,
            "objective_score": score_zw_max,
            "selection_score": selection_func(A, x_zw_max),
        },
        "disassociative_Zw_min": {
            "labels": x_zw_min,
            "objective_score": score_zw_min,
            "selection_score": selection_func(A, x_zw_min),
        },
        "core_periphery_Zd": {
            "labels": x_zd,
            "objective_score": score_zd,
            "selection_score": selection_func(A, x_zd),
        },
    }

    selected_name = max(
        candidates.keys(),
        key=lambda name: candidates[name]["selection_score"],
    )

    return {
        "selected_type": selected_name,
        "selected_labels": candidates[selected_name]["labels"],
        "selected_selection_score": candidates[selected_name]["selection_score"],
        "candidates": candidates,
    }