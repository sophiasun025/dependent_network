source('/home/msaa2017/Dependent_network/Functions/dependent_network_functions.R')
source('/home/msaa2017/submission_code_files/Functions/functions_KAP_CPD.R')

library(parallel)

# -----------------------------
# Settings
# -----------------------------
iter <- 100
T <- 100
tau <- 70
signal_list <- c(rep(0, 5), rep(0.05, 5))
B <- 1000

rho_list <- rep(c(0, 0.05, 0.2, 0.4, 0.8), 2)
p_within_before_list <- rep(0.3, length(rho_list))

ncores <- 10

method_specs <- list(
  KAP = list(type = "raw"),
  KAP_dependent = list(type = "lag_adjusted", kernel = "K_resid_mean", max_lag = Inf),
  KAP_dependent_lag6 = list(type = "lag_adjusted", kernel = "K_resid_mean", max_lag = 6),
  KAP_dependent_lag12 = list(type = "lag_adjusted", kernel = "K_resid_mean", max_lag = 12),
  KAP_dependent_lag6_adjust2 = list(type = "lag_adjusted", kernel = "K_resid_mean2", max_lag = 6)
)

make_kernel_pair <- function(K1_raw, K2_raw, spec) {
  if (spec$type == "raw") {
    return(list(K1 = K1_raw, K2 = K2_raw))
  }
  
  K1_lag_removed <- remove_lag_effect(K1_raw, max_lag = spec$max_lag)
  K2_lag_removed <- remove_lag_effect(K2_raw, max_lag = spec$max_lag)
  
  list(
    K1 = K1_lag_removed[[spec$kernel]],
    K2 = K2_lag_removed[[spec$kernel]]
  )
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
  
  max_S <- apply(S[, n0:n1, drop = FALSE], 1, max)
  min(1, sum(max_S >= max(scanZ$S$scan[n0:n1])) / B)
}

run_one_method <- function(K1_raw, K2_raw, spec, T, tau, B) {
  scanZ <- KAP_CPD_modified(K1_raw, K2_raw, spec, 0.5, 2, T)
  p_perm <- permpval_modified(T, K1_raw, K2_raw, spec, B = B)
  tauhat <- scanZ$S$tauhat
  
  list(
    pvalue = p_perm,
    tauhat = tauhat,
    detect = as.integer(p_perm <= 0.05),
    accurate = as.integer((p_perm <= 0.05) && (abs(tauhat - tau) <= 5))
  )
}

# -----------------------------
# One replicate
# -----------------------------
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
  
  K1_raw <- toy_data$K1
  K2_raw <- toy_data$K2
  
  method_results <- lapply(
    method_specs,
    function(spec) {
      run_one_method(K1_raw, K2_raw, spec, T = T, tau = tau, B = B)
    }
  )
  
  c(list(
    rep_id = rep_id,
    rho = rho,
    p_within_before = p_within_before,
    signal = signal
  ), method_results)
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
            accurate_count,
            detect_count / iter,
            accurate_count / iter
          ),
          paste0(
            method_name,
            c("_count", "_count_acc", "_power", "_acc_power")
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
  results_list,
  file.path(out_dir, "KAP_dependent_simulation_SBM_results_list_kernel_adjustment_comparison_lag6.rds")
)

saveRDS(
  summary_df,
  file.path(out_dir, "KAP_dependent_simulation_SBM_summary_df_kernel_adjustment_comparison_lag6.rds")
)
