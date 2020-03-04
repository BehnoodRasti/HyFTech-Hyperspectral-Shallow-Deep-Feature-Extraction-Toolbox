function [oa, pa, K, CM] = USFE_MSTV(HSI, Tr, Te, dim, Trees)
[FE_MSTV]= MSTV_Xu(HSI, dim);
[acc_Mean,acc_std,CM]=RF_ntimes_overal(FE_MSTV,Tr,Te,Trees);
pa=acc_Mean(1:dim,1);
oa=acc_Mean(dim+2,1);
K=acc_Mean(dim+3,1);