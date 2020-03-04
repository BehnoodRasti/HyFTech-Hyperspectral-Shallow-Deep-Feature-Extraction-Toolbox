% addpath('data');
% addpath(genpath('MSTV'));
% addpath(genpath('LPP'));
% addpath(genpath('OTVCA_V3'));
% addpath('RF classifier');

%% Load data
load Houston.mat;
HSI = Houston;
load TRLabel
load TSLabel
Tr = TRLabel;
Te = TSLabel;
clear TRLabel TSLabel Houston
%% Parameter setting
dim=max(unique(Te)); % Number of Features is set to the number of classes
Trees = 200; % for RF training
%% PCA+RF
[oa, pa, K, CM] = USFE_PCA(HSI, Tr, Te, dim, Trees);
aa = mean(pa);
%% MSTV+RF
[oa, pa, K, CM] = USFE_MSTV(HSI, Tr, Te, dim, Trees);
aa = mean(pa);
%% OTVCA Rasti et al.
[oa, pa, K, CM] = USFE_OTVCA(HSI, Tr, Te, dim, Trees);
aa = mean(pa);
%% LPP
[oa, pa, K, CM] = USFE_LPP(HSI, Tr, Te, dim, Trees);
aa = mean(pa);



