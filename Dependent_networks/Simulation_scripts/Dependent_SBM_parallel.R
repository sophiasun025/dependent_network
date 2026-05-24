source("~/networkcp/Dependent_networks/Functions/dependent_network_functions.R")
source("~/networkcp/submission_code_files/Functions/functions_KAP_CPD.R")

library(parallel)

# -----------------------------
# Settings
# -----------------------------
iter <- 100
T <- 100
tau <- 50
signal <- 0.05
B <- 1000

rho_list <- c(0, 0.2, 0.6, 0.8, 0.9,
              0, 0.2, 0.6, 0.8, 0.9)

p_within_before_list <- c(0.3, 0.3, 0.3, 0.3, 0.3,
                          0.5, 0.5, 0.5, 0.5, 0.5)

ncores <- 10

# -----------------------------
# One replicate
# -----------------------------
run_one_rep <- function(rep_id, rho, p_within_before,
                        T = 100, tau = 50, signal = 0.05, B = 1000) {
  
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
  
  K1 <- toy_data$K1
  K2 <- toy_data$K2
  
  # -----------------------------
  # KAP original
  # -----------------------------
  scanZ <- KAP_CPD_statistic(K1, K2, 0.5, 2, T)
  p_perm <- permpval2(T, K1, K2, B = B)
  KAP_tau <- scanZ$S$tauhat
  
  KAP_detect <- as.integer(p_perm <= 0.05)
  KAP_acc <- as.integer((p_perm <= 0.05) && (abs(KAP_tau - tau) <= 5))
  
  # -----------------------------
  # KAP dependent / lag removal
  # -----------------------------
  scanZ_dep <- KAP_CPD_dependent(K1, K2, 0.5, 2, T)
  p_perm_dep <- permpval_dependent(T, K1, K2, B = B)
  KAP_dep_tau <- scanZ_dep$S$tauhat
  
  KAP_dep_detect <- as.integer(p_perm_dep <= 0.05)
  KAP_dep_acc <- as.integer((p_perm_dep <= 0.05) && (abs(KAP_dep_tau - tau) <= 5))
  
  list(
    rep_id = rep_id,
    rho = rho,
    p_within_before = p_within_before,
    
    KAP = list(
      pvalue = p_perm,
      tauhat = KAP_tau,
      detect = KAP_detect,
      accurate = KAP_acc
    ),
    
    KAP_dependent = list(
      pvalue = p_perm_dep,
      tauhat = KAP_dep_tau,
      detect = KAP_dep_detect,
      accurate = KAP_dep_acc
    )
  )
}

# -----------------------------
# One setting: fixed rho and p_within
# -----------------------------
run_one_setting <- function(setting_id) {
  
  rho <- rho_list[setting_id]
  p_within_before <- p_within_before_list[setting_id]
  
  cat("\n[INFO] Running setting", setting_id,
      "rho =", rho,
      "p_within_before =", p_within_before, "\n")
  
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
  
  KAP_count <- sum(sapply(reps, function(x) x$KAP$detect))
  KAP_count_acc <- sum(sapply(reps, function(x) x$KAP$accurate))
  
  KAP_dep_count <- sum(sapply(reps, function(x) x$KAP_dependent$detect))
  KAP_dep_count_acc <- sum(sapply(reps, function(x) x$KAP_dependent$accurate))
  
  summary <- data.frame(
    setting_id = setting_id,
    rho = rho,
    p_within_before = p_within_before,
    signal = signal,
    iter = iter,
    
    KAP_count = KAP_count,
    KAP_count_acc = KAP_count_acc,
    KAP_power = KAP_count / iter,
    KAP_acc_power = KAP_count_acc / iter,
    
    KAP_dependent_count = KAP_dep_count,
    KAP_dependent_count_acc = KAP_dep_count_acc,
    KAP_dependent_power = KAP_dep_count / iter,
    KAP_dependent_acc_power = KAP_dep_count_acc / iter
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

# Combine summaries into one data frame
summary_df <- do.call(
  rbind,
  lapply(results_list, function(x) x$summary)
)

print(summary_df)

# Save output
saveRDS(results_list, "KAP_dependent_simulation_SBM_results_list.rds")
saveRDS(summary_df, "KAP_dependent_simulation_SBM_summary_df.rds")