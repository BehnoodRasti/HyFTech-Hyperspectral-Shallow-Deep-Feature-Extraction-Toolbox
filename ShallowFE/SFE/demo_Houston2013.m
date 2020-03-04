clc;
clear;
close all;

addpath('data');
addpath('funcitons');

addpath('LDA');
addpath('CGDA');
addpath('LSDR');
addpath('J-Play');
addpath('RF classifier');

%% Load data
load Houston.mat;
HSI = Houston;
load TRLabel
load TSLabel
Tr = TRLabel;
Te = TSLabel;
clear TRLabel TSLabel Houston
%% Parameter setting
dim = 15; % for Houston2013
Trees = 200; % for RF training

%% LDA
[oa, pa, K, CM] = SFE_LDA(HSI, Tr, Te, dim, Trees);
aa = mean(pa);

%% CGDA
lamda = 0.1;
[oa, pa, K, CM] = SFE_CGDA(HSI, Tr, Te, dim, lamda, Trees);
aa = mean(pa);

%% LSDR
[oa, pa, K, CM] = SFE_LSDR(HSI, Tr, Te, dim, Trees);
aa = mean(pa);

%% JPLAY
num = 10;
sigma = 0.1;
alfa = 1;
beta = 0.1;
gamma = 0.1;
rho = 2;
maxiter = 1000;
eta = 1;
epsilon = 1e-4;
[oa, pa, K, CM] = SFE_JPLAY(HSI, Tr, Te, dim, num, sigma, alfa, beta, gamma, rho, maxiter, eta, epsilon, Trees);
 