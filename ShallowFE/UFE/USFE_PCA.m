function [oa, pa, K, CM] = USFE_PCA(HSI, Tr, Te, dim, Trees)
[nx,ny,nz]=size(HSI);
data=reshape(HSI,nx*ny,nz);
clear HSI
%% matlab PCA from the stats toolbox (C:\Program Files\MATLAB\R2018b\toolbox\stats\stats\pca.m)
[code] = pca(data');
FE_Mpca=reshape(code(:,1:dim),nx,ny,dim);
[acc_Mean,acc_std,CM]=RF_ntimes_overal(FE_Mpca,Tr,Te,Trees);
pa=acc_Mean(1:dim,1);
oa=acc_Mean(dim+2,1);
K=acc_Mean(dim+3,1);