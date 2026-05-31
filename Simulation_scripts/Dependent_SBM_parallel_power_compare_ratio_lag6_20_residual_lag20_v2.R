source('/home/msaa2017/Dependent_network/Functions/dependent_network_functions.R')
source('/home/msaa2017/submission_code_files/Functions/functions_KAP_CPD.R')

library(parallel)

# -----------------------------
# Settings
# -----------------------------
iter <- 100
T <- 100
tau <- 70
signal_list <- c(rep(0, 6), rep(0.05, 6))
B <- 1000

rho_list <- rep(c(0, 0.05, 0.1, 0.2, 0.4, 0.5), 2)
p_within_before_list <- rep(0.3, length(rho_list))

ncores <- 15

method_specs <- list(
  KAP = list(type = "raw"),
  KAP_dependent_raw_mean_lag20 = list(type = "lag_adjusted", kernel = "K_resid_raw_mean", max_lag = 20),
  KAP_dependent_raw_mean_smooth_lag20 = list(type = "lag_adjusted", kernel = "K_resid_raw_mean_smooth", max_lag = 20),
  KAP_dependent_ratio_mean = list(type = "lag_adjusted", kernel = "K_ratio_mean", max_lag = Inf),
  KAP_dependent_ratio_mean_lag6 = list(type = "lag_adjusted", kernel = "K_ratio_mean", max_lag = 6),
  KAP_dependent_ratio_mean_lag20 = list(type = "lag_adjusted", kernel = "K_ratio_mean", max_lag = 20)
)


################################################Functions

get_method_kernels <- function(K1_raw, K2_raw, method_specs) {
  kernels_by_method <- list()
  
  # Raw methods
  raw_methods <- names(method_specs)[
    vapply(method_specs, function(x) x$type == "raw", logical(1))
  ]
  
  for (method_name in raw_methods) {
    kernels_by_method[[method_name]] <- list(K1 = K1_raw, K2 = K2_raw)
  }
  
  # Lag-adjusted methods, grouped by max_lag
  lag_methods <- names(method_specs)[
    vapply(method_specs, function(x) x$type == "lag_adjusted", logical(1))
  ]
  
  max_lags <- unique(vapply(
    method_specs[lag_methods],
    function(x) as.character(x$max_lag),
    character(1)
  ))
  
  for (max_lag_chr in max_lags) {
    max_lag <- if (max_lag_chr == "Inf") Inf else as.numeric(max_lag_chr)
    
    K1_lag_removed <- remove_lag_effect(K1_raw, max_lag = Inf)
    K2_lag_removed <- remove_lag_effect(K2_raw, max_lag = Inf)
    
    these_methods <- lag_methods[
      vapply(
        method_specs[lag_methods],
        function(x) as.character(x$max_lag) == max_lag_chr,
        logical(1)
      )
    ]
    
    for (method_name in these_methods) {
      kernel_name <- method_specs[[method_name]]$kernel
      
      if (is.null(K1_lag_removed[[kernel_name]]) ||
          is.null(K2_lag_removed[[kernel_name]])) {
        stop("Kernel not found in remove_lag_effect output: ", kernel_name)
      }
      
      kernels_by_method[[method_name]] <- list(
        K1 = K1_lag_removed[[kernel_name]],
        K2 = K2_lag_removed[[kernel_name]]
      )
    }
  }
  
  kernels_by_method[names(method_specs)]
}

run_all_methods_one_rep <- function(K1_raw, K2_raw, method_specs, T, tau, B,
                                    n0 = ceiling(0.05 * T),
                                    n1 = floor(0.95 * T)) {
  if (n0 < 2) {
    n0 <- 2
  }
  if (n1 > (T - 2)) {
    n1 <- T - 2
  }
  
  method_names <- names(method_specs)
  
  # Observed kernels and observed statistics
  obs_kernels <- get_method_kernels(K1_raw, K2_raw, method_specs)
  
  obs_stats <- lapply(method_names, function(method_name) {
    K <- obs_kernels[[method_name]]
    KAP_CPD_statistic(K$K1, K$K2, 0.5, 2, T, n0, n1)
  })
  names(obs_stats) <- method_names
  
  obs_max <- vapply(
    obs_stats,
    function(x) max(x$S$scan[n0:n1]),
    numeric(1)
  )
  
  perm_max <- matrix(
    0,
    nrow = B,
    ncol = length(method_names),
    dimnames = list(NULL, method_names)
  )
  
  for (b in 1:B) {
    if (b %% 1000 == 0) {
      message(b, " permutations completed.\n")
    }
    
    id0 <- sample(1:T, replace = FALSE)
    K1_id <- K1_raw[id0, id0]
    K2_id <- K2_raw[id0, id0]
    
    # Important: recompute lag removal after permutation
    perm_kernels <- get_method_kernels(K1_id, K2_id, method_specs)
    
    for (method_name in method_names) {
      K <- perm_kernels[[method_name]]
      Sstar <- calculate_statistic_new(K$K1, K$K2, T, n0, n1)
      perm_max[b, method_name] <- max(Sstar$S[n0:n1])
    }
  }
  
  out <- lapply(method_names, function(method_name) {
    p_perm <- min(1, sum(perm_max[, method_name] >= obs_max[method_name]) / B)
    tauhat <- obs_stats[[method_name]]$S$tauhat
    
    list(
      pvalue = p_perm,
      tauhat = tauhat,
      detect = as.integer(p_perm <= 0.05),
      accurate = as.integer((p_perm <= 0.05) && (abs(tauhat - tau) <= T * 0.05))
    )
  })
  names(out) <- method_names
  out
}
p_within_before=0.3
signal=0.5
run_one_rep <- function(rep_id, rho, p_within_before,
                        T = 100, tau = 70, signal = 0.05, B = 1000) {
  set.seed(10000 + rep_id)

  toy_data <- generate_dependent_sbm_cp(
    rho = rho,
    p_within_before = p_within_before,
    p_across_before = 0.2,
    p_within_after = p_within_before + signal,
    p_across_after = 0.2,
    block_nums = c(10, 10, 10, 10, 10),
    tau = tau,
    T = T
  )
  
  method_results <- run_all_methods_one_rep(
    K1_raw = toy_data$K1,
    K2_raw = toy_data$K2,
    method_specs = method_specs,
    T = T,
    tau = tau,
    B = B
  )
  
  c(
    list(
      rep_id = rep_id,
      rho = rho,
      p_within_before = p_within_before,
      signal = signal
    ),
    method_results
  )
}


# -----------------------------
# One setting: fixed rho, p_within, and signal
# -----------------------------
run_one_setting <- function(setting_id) {
  
  rho <- rho_list[setting_id]
  p_within_before <- p_within_before_list[setting_id]
  signal <- signal_list[setting_id]
  
  cat("\n[INFO] Running setting", setting_id,
      "rho =", rho,
      "p_within_before =", p_within_before,
      "signal =", signal, "\n")
  
  reps <- mclapply(
    1:iter,
    function(i) {
      run_one_rep(
        rep_id = i,
        rho = rho,
        p_within_before = p_within_before,
        T = T,
        tau = tau,
        signal = signal,
        B = B
      )
    },
    mc.cores = ncores
  )
  
  failed <- vapply(reps, inherits, logical(1), what = "try-error")
  if (any(failed)) {
    stop("Replicate failures in setting ", setting_id, ": ", reps[[which(failed)[1]]])
  }
  
  method_summary <- unlist(
    lapply(
      names(method_specs),
      function(method_name) {
        detect_count <- sum(sapply(reps, function(x) x[[method_name]]$detect))
        accurate_count <- sum(sapply(reps, function(x) x[[method_name]]$accurate))
        
        setNames(
          c(
            detect_count,
            accurate_count
          ),
          paste0(
            method_name,
            c("_count", "_count_acc")
          )
        )
      }
    )
  )
  method_summary <- as.list(method_summary)
  
  summary <- data.frame(
    setting_id = setting_id,
    rho = rho,
    p_within_before = p_within_before,
    signal = signal,
    iter = iter,
    method_summary,
    check.names = FALSE
  )
  
  list(
    setting = list(
      setting_id = setting_id,
      rho = rho,
      p_within_before = p_within_before,
      signal = signal,
      T = T,
      tau = tau,
      B = B,
      iter = iter
    ),
    summary = summary,
    reps = reps
  )
}

# -----------------------------
# Run all settings
# -----------------------------
results_list <- lapply(seq_along(rho_list), run_one_setting)

summary_df <- do.call(
  rbind,
  lapply(results_list, function(x) x$summary)
)

print(summary_df)

# Save output
out_dir <- "/home/msaa2017/RDS/Dependent_network"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}


saveRDS(
  summary_df,
  file.path(out_dir, "KAP_dependent_SBM_summary_df_kernel_adjustment_comparison_ratio2_lag6_20_residual_lag20_v2.rds")
)

