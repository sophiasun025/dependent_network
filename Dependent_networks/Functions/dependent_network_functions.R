library(igraph)


generate_dependent_sbm_cp <- function(
    rho,
    p_within_before,
    p_across_before,
    p_within_after,
    p_across_after,
    block_nums,
    tau,
    T
) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("Package 'igraph' is required. Please install it using install.packages('igraph').")
  }
  
  N <- sum(block_nums)
  community <- rep(seq_along(block_nums), times = block_nums)
  
  if (tau < 2 || tau >= T-2) {
    stop("tau should satisfy 1 <= tau < T.")
  }
  
  # Helper function to construct SBM probability matrix
  make_E <- function(p_within, p_across) {
    E <- matrix(p_across, nrow = N, ncol = N)
    E[outer(community, community, "==")] <- p_within
    diag(E) <- 0
    return(E)
  }
  
  # Before and after change point probability matrices
  E_before <- make_E(p_within_before, p_across_before)
  E_after  <- make_E(p_within_after,  p_across_after)
  adj_list <- vector("list", T)
  upper_idx <- upper.tri(matrix(0, N, N))
  
  # Initial graph: G_1 ~ SBM(E_before)
  A <- matrix(0, nrow = N, ncol = N)
  A[upper_idx] <- rbinom(
    sum(upper_idx),
    size = 1,
    prob = E_before[upper_idx]
  )
  A <- A + t(A)
  adj_list[[1]] <- A
  
  # Dynamic generation
  for (t in 1:(T - 1)) {
    A_old <- adj_list[[t]]
    A_new <- matrix(0, nrow = N, ncol = N)
    
    # Use E_before for transitions up to tau,
    # and E_after after tau.
    if (t < tau) {
      E_current <- E_before
    } else {
      E_current <- E_after
    }
    
    prob_mat <- matrix(0, nrow = N, ncol = N)
    
    # If y_ij^t = 1
    prob_mat[A_old == 1] <- rho * (1 - E_current[A_old == 1]) +
      E_current[A_old == 1]
    
    # If y_ij^t = 0
    prob_mat[A_old == 0] <- (1 - rho) * E_current[A_old == 0]
    
    diag(prob_mat) <- 0
    
    A_new[upper_idx] <- rbinom(
      sum(upper_idx),
      size = 1,
      prob = prob_mat[upper_idx]
    )
    A_new <- A_new + t(A_new)
    
    adj_list[[t + 1]] <- A_new
  }
  
  # Flatten upper triangular part of each adjacency matrix
  adj_matrix_flat <- do.call(rbind, lapply(adj_list, function(A) {
    A[upper_idx]
  }))
  
  edge_names <- which(upper_idx, arr.ind = TRUE)
  colnames(adj_matrix_flat) <- paste0(
    "edge_",
    edge_names[, 1],
    "_",
    edge_names[, 2]
  )
  
  # Convert adjacency matrices to igraph objects
  graph_list <- lapply(adj_list, function(A) {
    igraph::graph_from_adjacency_matrix(
      A,
      mode = "undirected",
      diag = FALSE
    )
  })
  
  K1 <- gaussiankernel(adj_matrix_flat)
  K2 <- CalculateGraphletKernel(graph_list, 3)
  diag(K2) <- 0
  return(list(
    K1=K1,
    K2=K2,
    adj_list = adj_list,
    graph_list = graph_list,
    adj_matrix_flat = adj_matrix_flat,
    E_before = E_before,
    E_after = E_after,
    community = community,
    tau = tau,
    rho = rho
  ))
}

#@ Lag effect removal
remove_global_lag_effect_test <- function(K,max_lag,include_diag = FALSE) {
  n <- nrow(K)
  K_work <- K
  
  # ignore diagonal
  if (!include_diag) {
    diag(K_work) <- NA
  }
  
  # lag_mat[i, j] = |i - j|
  lag_mat <- abs(row(K_work) - col(K_work))
  
  # Median kernel value for each lag h
  lag_mean <- tapply(
    as.vector(K_work),
    as.vector(lag_mat),
    median,
    na.rm = TRUE
  )
  
  # Matrix whose (i,j) entry is lag_mean[|i-j|]
  # adjust_mat <- matrix(
  #   lag_mean[as.character(lag_mat)],
  #   nrow = n,
  #   ncol = n
  # )
  adjust_mat=mean(K)
  
  # Remove lag effect
  K_resid <- K - adjust_mat
  
  # If diagonal was ignored, set residual diagonal to 0
  if (!include_diag) {
    diag(K_resid) <- 0
  }
  
  # Enforce symmetry
  K_resid <- (K_resid + t(K_resid)) / 2
  
  list(
    K_resid = K_resid,
    lag_mean = lag_mean
  )
}


remove_lag_effect <- function(K, max_lag = Inf, include_diag = FALSE) {
  n <- nrow(K)
  K_work <- K
  
  # Usually ignore diagonal because K_ii is self-similarity
  if (!include_diag) {
    diag(K_work) <- NA
  }
  
  # lag_mat[i, j] = |i - j|
  lag_mat <- abs(row(K_work) - col(K_work))
  
  # Mean / median / sd kernel value for each lag h
  lag_mean <- tapply(
    as.vector(K_work),
    as.vector(lag_mat),
    mean,
    na.rm = TRUE
  )
  
  lag_median <- tapply(
    as.vector(K_work),
    as.vector(lag_mat),
    median,
    na.rm = TRUE
  )
  
  lag_sd <- tapply(
    as.vector(K_work),
    as.vector(lag_mat),
    sd,
    na.rm = TRUE
  )
  
  lag_median_median <- median(lag_median, na.rm = TRUE)
  lag_mean_mean <- mean(lag_mean, na.rm = TRUE)
  
  # Lag-specific adjustment values
  lag_adjust_mean <- lag_mean / lag_mean_mean * lag_sd
  lag_adjust_mean2 <- (lag_mean - lag_mean_mean) / lag_sd
  lag_adjust_median <- lag_median / lag_median_median * lag_sd
  lag_adjust_mean[!is.finite(lag_adjust_mean)] <- 0
  lag_adjust_mean2[!is.finite(lag_adjust_mean2)] <- 0
  lag_adjust_median[!is.finite(lag_adjust_median)] <- 0
  
  # Build full adjustment matrices
  adjust_mat_mean <- matrix(
    lag_adjust_mean[as.character(lag_mat)],
    nrow = n,
    ncol = n
  )
  
  adjust_mat_mean2 <- matrix(
    lag_adjust_mean2[as.character(lag_mat)],
    nrow = n,
    ncol = n
  )
  
  adjust_mat_median <- matrix(
    lag_adjust_median[as.character(lag_mat)],
    nrow = n,
    ncol = n
  )
  
  # Only adjust lags up to max_lag
  use_idx <- lag_mat <= max_lag
  
  # If not including diagonal, do not adjust diagonal
  if (!include_diag) {
    use_idx[lag_mat == 0] <- FALSE
  }
  
  # Entries with lag > max_lag receive zero adjustment
  adjust_mat_mean[!use_idx] <- 0
  adjust_mat_mean2[!use_idx] <- 0
  adjust_mat_median[!use_idx] <- 0
  
  # Remove lag effect
  K_resid_mean <- K - adjust_mat_mean
  K_resid_mean2 <- K - adjust_mat_mean2
  K_resid_median <- K - adjust_mat_median
  
  # If diagonal was ignored, set residual diagonal to 0
  if (!include_diag) {
    diag(K_resid_mean) <- 0
    diag(K_resid_mean2) <- 0
    diag(K_resid_median) <- 0
  }
  
  # Enforce symmetry
  K_resid_mean <- (K_resid_mean + t(K_resid_mean)) / 2
  K_resid_mean2 <- (K_resid_mean2 + t(K_resid_mean2)) / 2
  K_resid_median <- (K_resid_median + t(K_resid_median)) / 2
  
  list(
    K_resid_mean = K_resid_mean,
    K_resid_mean2 = K_resid_mean2,
    K_resid_median = K_resid_median,
    lag_mean = lag_mean,
    lag_median = lag_median,
    lag_sd = lag_sd,
    lag_adjust_mean = lag_adjust_mean,
    lag_adjust_mean2 = lag_adjust_mean2,
    lag_adjust_median = lag_adjust_median,
    max_lag = max_lag
  )
}

plot_matrix_heatmap <- function(M, title = "Kernel Matrix Heat Map") {
  df <- as.data.frame(as.table(M))
  colnames(df) <- c("Row", "Col", "Value")
  
  df$Row <- as.numeric(df$Row)
  df$Col <- as.numeric(df$Col)
  
  ggplot(df, aes(x = Col, y = Row, fill = Value)) +
    geom_tile() +
    scale_x_continuous(
      breaks = seq(1, ncol(M), by = 10),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = seq(1, ncol(M), by = 10),
      expand = c(0, 0)
    ) +
    labs(
      title = title,
      x='Row',
      y='Column'
    ) +
    theme_minimal()
}

make_kernel_pair <- function(K1_raw, K2_raw, spec) {
  if (spec$type == "raw") {
    return(list(K1 = K1_raw, K2 = K2_raw))
  }
  
  K1_adjusted <- remove_lag_effect(K1_raw, max_lag = spec$max_lag)[[spec$kernel]]
  K2_adjusted <- remove_lag_effect(K2_raw, max_lag = spec$max_lag)[[spec$kernel]]
  
  list(K1 = K1_adjusted, K2 = K2_adjusted)
}

KAP_CPD_modified <- function(K1_raw, K2_raw, spec, r1 = 0.5, r2 = 2, n,
                             n0 = ceiling(0.05 * n), n1 = floor(0.95 * n)) {
  kernels <- make_kernel_pair(K1_raw, K2_raw, spec)
  KAP_CPD_statistic(kernels$K1, kernels$K2, r1, r2, n, n0, n1)
}

permpval_modified <- function(n, K1_raw, K2_raw, spec, B = 1000,
                              n0 = ceiling(0.05 * n), n1 = floor(0.95 * n)) {
  if (n0 < 2) {
    n0 <- 2
  }
  if (n1 > (n - 2)) {
    n1 <- n - 2
  }
  
  S <- matrix(0, B, n)
  scanZ <- KAP_CPD_modified(K1_raw, K2_raw, spec, n = n, n0 = n0, n1 = n1)
  
  for (b in 1:B) {
    if (b %% 1000 == 0) {
      message(b, " permutations completed.\n")
    }
    
    id0 <- sample(1:n, replace = FALSE)
    K1_id <- K1_raw[id0, id0]
    K2_id <- K2_raw[id0, id0]
    Sstar <- KAP_CPD_modified(K1_id, K2_id, spec, n = n, n0 = n0, n1 = n1)
    
    S[b, ] <- Sstar$S$scan
  }
  
  max_S <- apply(S[, n0:n1], 1, max)
  max_S_sort <- sort(max_S)
  min(1, length(which(max_S_sort >= max(scanZ$S$scan[n0:n1]))) / B)
}



# par(mfrow=c(1,2))
# plot(r$lag_adjust_median,ylim=c(0,0.03))
# plot(r$lag_adjust_mean,ylim=c(0.0,0.03))
