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


remove_lag_effect <- function(K, max_lag = Inf, s = 0.6) {
  # Returns lag-adjusted kernel matrices:
  # K_resid_raw_mean, K_resid_mean_mm, K_resid_smooth_mean,
  # K_ratio_median, K_ratio_median_smooth
  n <- nrow(K)
  K_work <- K
  diag(K_work) <- NA
  
  #Create lags related statistics
  lag_mat <- abs(row(K_work) - col(K_work))
  
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
  lag_mean_mean <- mean(lag_mean, na.rm = TRUE)
  lag_mean_median <- median(lag_mean, na.rm = TRUE)
  lag_median_median <- median(lag_median, na.rm = TRUE)
  lag_mean_sd=sd(lag_mean, na.rm = TRUE)
  lag_median_sd=sd(lag_median, na.rm = TRUE)
  
  #Function that fits a smoothing spline
  smooth_adjustment <- function(lag_mean) {
    lags <- as.numeric(names(lag_mean))
    valid <- is.finite(lags) & is.finite(lag_mean)
    if (sum(valid) < 4 || length(unique(lags[valid])) < 4) {
      return(lag_mean)
    }
    fit <- smooth.spline(
      x = lags[valid],
      y = as.numeric(lag_mean[valid]),
      spar = s
    )
    x_smooth <- predict(fit, x = lags)$y
    names(x_smooth) <- names(lag_mean)
    x_smooth
  }
  
  
  
  #lag-specific adjustment families and convert the lags into matrices
  lag_adjust_raw_mean <- lag_mean
  lag_adjust_raw_mean_plus_mean_median <- lag_mean-lag_mean_median
  lag_adjust_raw_median <- lag_median
  lag_adjust_smooth_mean <- smooth_adjustment(lag_adjust_raw_mean)
  lag_adjust_smooth_median <- smooth_adjustment(lag_adjust_raw_median)
  
  build_adjustment_matrix <- function(lag_adjustment) {
    adjust_mat <- matrix(
      lag_adjustment[as.character(lag_mat)],
      nrow = n,
      ncol = n
    )
    adjust_mat[!is.finite(adjust_mat)] <- 0
    adjust_mat
  }
  
  #as adj matrix
  adjust_mat_raw_mean <- build_adjustment_matrix(lag_adjust_raw_mean)
  adjust_mat_raw_mean_plus_mean_median<-build_adjustment_matrix(lag_adjust_raw_mean_plus_mean_median)
  adjust_mat_raw_median <- build_adjustment_matrix(lag_adjust_raw_median)
  adjust_mat_smooth_mean <- build_adjustment_matrix(lag_adjust_smooth_mean)
  adjust_mat_smooth_median <- build_adjustment_matrix(lag_adjust_smooth_median)
  
  #only adjust up to max lag
  use_idx <- lag_mat <= max_lag
  use_idx[lag_mat == 0] <- FALSE
  adjust_mat_raw_mean[!use_idx] <- 0
  adjust_mat_smooth_mean[!use_idx] <- 0
  adjust_mat_raw_median[!use_idx] <- 0
  adjust_mat_smooth_median[!use_idx] <- 0
  adjust_mat_raw_mean_plus_mean_median[!use_idx] <- 0
  
  residual_kernel <- function(adjust_mat) {
    K_resid <- K - adjust_mat
    diag(K_resid) <- 0
    (K_resid + t(K_resid)) / 2
  }
  
  
  ratio_kernel <- function(adjust_mat,mm) {
    K_ratio <- K
    ratio_idx <- is.finite(adjust_mat) & adjust_mat != 0
    if (is.finite(mm)) {
      K_ratio[ratio_idx] <- K[ratio_idx] / adjust_mat[ratio_idx] * mm
    }
    (K_ratio + t(K_ratio)) / 2
  }
  
  #construct adjusted Kernel matrix
  #1,2
  K_resid_raw_mean <- residual_kernel(adjust_mat_raw_mean)
  K_resid_mean_mm <- residual_kernel(adjust_mat_raw_mean_plus_mean_median)
  K_resid_smooth_mean <- residual_kernel(adjust_mat_smooth_mean)
  #3,4
  # K_ratio_mean <- ratio_kernel(adjust_mat_raw_mean,lag_mean_mean)
  # K_ratio_mean_smooth <- ratio_kernel(adjust_mat_raw_mean_smooth,lag_mean_mean)
  #3,4
  K_ratio_median <- ratio_kernel(adjust_mat_raw_median,lag_median_median)
  K_ratio_median_smooth <- ratio_kernel(adjust_mat_smooth_median,lag_median_median)
  
  list(
    K_resid_raw_mean = K_resid_raw_mean,
    K_resid_mean_mm = K_resid_mean_mm,
    K_resid_smooth_mean = K_resid_smooth_mean,
    # K_ratio_mean=K_ratio_mean,
    # K_ratio_mean_smooth=K_ratio_mean_smooth,
    K_ratio_median=K_ratio_median,
    K_ratio_median_smooth=K_ratio_median_smooth,
    lag_mean = lag_mean,
    lag_median = lag_median,
    lag_mean_sd = lag_mean_sd,
    lag_adjust_smooth_mean = lag_adjust_smooth_mean,
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





# par(mfrow=c(1,2))
# plot(r$lag_adjust_median,ylim=c(0,0.03))
# plot(r$lag_adjust_mean,ylim=c(0.0,0.03))
