function [out, idxtrain,eigvector, eigvalue] = kpca(spectraldata, No_Train, dimension, kernel, parameter)

%
%  Kernel principal component analysis, KPCA
%
% [out, eigvector, eigvalue] = kpca(spectraldata, No_Train, dimension, kernel, parameter)
%
% Input
%
% spectraldata       - input hyperspectral data with 3-D, ncols by nrows by nbands
% No_Train           - randomly selected samples for training (<= 5000),
%                       depending on the memory of your PC
% dimension          - the dimension of extracted kernelized PCs (with largest eigen values)
% kernel             - kernel function
% parameter          - parameters for kernel fuction
%                        - kernel = 'linear';% linear kernel
%                        - kernel = 'Gaussian'; parameter=1; %Gausian kernel
%                        - kernel = 'poly'; parameter=[1,3];% third order polynomial 
%
%
% Output
%
% out                - the extracted kernelized PCs
% eigvector          - the eigenvectors divided by square root of corresponding eigenvalues
% eigvalue           - the first dimension largest eigenvalues

if nargin <  3, error('not enough input'); end
if nargin <  4
    if strncmp(kernel,'Gaussian',1)
        parameter=1;
    elseif strncmp(kernel,'poly',1) 
        parameter=[1, 3];
    end
end
    
[nrows,ncols,nbands] = size(spectraldata);
X0 = reshape(spectraldata,nrows*ncols,nbands);
clear spectraldata

%% sub-sample for training, select No_Train samples randomly
rand('state',4711007);% initialization of rand
if No_Train>nrows*ncols, No_Train = nrows*ncols; end
idxtrain = randsample(nrows*ncols, No_Train);
X = double(X0(idxtrain,:));
ntrain = size(X,1);

Xtest = X0;
ntest = size(Xtest,1);
clear X0;

%% kernelized training data and centering the kernel matrix
[K scale sums] = kernelize_training(kernel, X, parameter);
meanK = mean(K(:));
meanrowsK = mean(K);
K = centering(K);

%% select the first dimension eigvectors
dimout = ntrain;
if dimout>dimension, dimout=dimension; end
[eigvector,eigvalue,flagk] = eigs(K, dimout, 'LM');

if flagk~=0, warning('*** Convergence problems in eigs ***'); end
eigvalue = diag(abs(eigvalue))';
eigvector = sqrt(ntrain-1)*eigvector*diag(1./sqrt(eigvalue));
clear K

out = NaN(ntest,dimout);
%% kernelized the test samples, center kernel and calculate kernel PCs
for rr=1:nrows
     idx = (rr-1)*ncols+1;
     idx = idx:(idx+ncols-1);
     Xk = kernelize_test(kernel, X, Xtest(idx,:), scale, sums);
     Xk = Xk - repmat(meanrowsK,ncols,1)' - repmat(mean(Xk),ntrain,1) + meanK;
     out(idx,:) = Xk'*eigvector;
end

out = reshape(out,nrows,ncols,dimout);
