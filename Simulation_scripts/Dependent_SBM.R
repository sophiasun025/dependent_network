source('~/networkcp/Dependent_networks/Functions/dependent_network_functions.R')
source('~/networkcp/submission_code_files/Functions/functions_KAP_CPD.R')
iter=100
count_KAP=count_KAP_acc=0
count_KAP_dependent=count_KAP_dependent_acc=0
T=100
tau=50
signal=0.05
rho_list=c(0,0.2,0.6,0.8,0.9,0,0.2,0.6,0.8,0.9)
p_within_before_list=c(0.3,0.3,0.3,0.3,0.3,0.5,0.5,0.5,0.5,0.5)

for (j in 1:len(rho_list)){
  for (i in 1:iter){
    toy_data=generate_dependent_sbm_cp(
      rho=rho_list[j],
      p_within_before=p_within_before_list[j],
      p_across_before=0.2,
      p_within_after=p_within_before+signal,
      p_across_after=0.2,
      block_nums=c(10,10,10,10,10),
      tau=tau,
      T=T
    )
    K1=toy_data$K1
    K2=toy_data$K2
    #KAP Original
    scanZ <- KAP_CPD_statistic(K1, K2, 0.5, 2, T)
    p_perm <- permpval2(T, K1, K2, B = 1000)
    KAP_tau <- scanZ$S$tauhat
    print(p_perm)
    print(KAP_tau)
    if (p_perm<=0.05){
      count_KAP=count_KAP+1
      if (abs(KAP_tau-tau)<=5){
        count_KAP_acc=count_KAP_acc+1
      }
    }
    
    #KAP Lag removal
    
    scanZ2 <- KAP_CPD_dependent(K1, K2, 0.5, 2, T)
    p_perm2 <- permpval_dependent(T, K1, K2, B = 1000)
    KAP_dependent_tau <- scanZ2$S$tauhat
    print(p_perm2)
    print(KAP_dependent_tau)
    if (p_perm2<=0.05){
      count_KAP_dependent=count_KAP_dependent+1
      if (abs(KAP_dependent_tau-tau)<=5){
        count_KAP_dependent_acc=count_KAP_dependent_acc+1
      }
    }
  }
  print('KAP')
  print(count_KAP)
  print(count_KAP_acc)
  
  print('KAP dependent')
  print(count_KAP_dependent)
  print(count_KAP_dependent_acc)
}