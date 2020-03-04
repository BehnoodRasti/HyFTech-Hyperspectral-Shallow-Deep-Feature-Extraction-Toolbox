function K = kernelize_test(kernel, Xtrain, Xtest, parameter, Xtrainsum)

%
% K = kernelize_test(kernel, Xtrain, Xtest, parameter, Xtrainsum);
%
% Kernelize the test data with training data
%
% Input:
%
% kernel    - kernel type, 'Gaussian' etc.
% Xtrain    - training data
% Xtest     - new test data to be kernelized with training data
% parameter - scale parameter for kernel
% Xtrainsum - aux variable for training data
%             (row or columns sum for symmetric kernel matrix)
%
% Output:
%
% K         - kernel matrix for test data with training data
%
%
% Copyright 2009-2011
% Wenzhi Liao
% wliao@telin.ugent.be, http://telin.ugent.be/~wliao/
% 2 Oct 2010

if nargin<4, error('not enough input'); end

[ntrain dimtrain] = size(Xtrain);
[ntest dimtest] = size(Xtest);
if dimtrain~=dimtest, error('Xtrain and Xtest must have same dimension'); end

if ~strncmp(kernel,'linear',1)
    if nargin<5, Xtrainsum = sum(Xtrain.*Xtrain,2); end
    Xtestsum = sum(Xtest.*Xtest,2);
    K0 = repmat(Xtestsum',ntrain,1);
    Ki = repmat(Xtrainsum',ntest,1); 
    K = K0 + Ki' - 2*Xtrain*Xtest';
    clear K0 Ki
end

switch kernel    
    case 'linear'
        K  = Xtrain*Xtest'; % linear kernel
    case 'Gaussian'
        sigma2 = 2*parameter^2;
        K = exp(-K/sigma2); % Gaussian
    case 'poly'
        param1 = parameter(1); param2 = parameter(2);
        K = (K + param1) .^ param2; % poly kernel
    otherwise
        error('Unknown kernel function.');
end
