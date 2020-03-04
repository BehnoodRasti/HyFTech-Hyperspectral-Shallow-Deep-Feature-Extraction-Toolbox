function [oa, pa, K, CM] = USFE_OTVCA(HSI, Tr, Te, dim, Trees)
[FE_OTVCA]=OTVCA_V3(HSI,dim);
[acc_Mean,acc_std,CM]=RF_ntimes_overal(FE_OTVCA,Tr,Te,Trees);
pa=acc_Mean(1:dim,1);
oa=acc_Mean(dim+2,1);
K=acc_Mean(dim+3,1);