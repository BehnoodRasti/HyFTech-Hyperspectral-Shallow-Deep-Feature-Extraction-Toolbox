function [K, scale, Xtrainsum] = kernelize_training(kernel, X, parameter)

%
% [K scale Xtrainsum] = kernelize_training(kernel, X, parameter);
%
% Kernelize the training data
%
% Input
% kernel       - kernel type, 'Gaussian' etc.
% X            - training data
% parameter    - parameters for kernel
%
% Output
% K         - kernel for training data
% scale     - scale parameter for kernel testing
% Xtrainsum - row or columns sum for symmetric kernel matrix (symmetric)
%
%
% Copyright 2009-2011
% Wenzhi Liao
% wliao@telin.ugent.be, http://telin.ugent.be/~wliao/
% 2 Oct 2010


if nargin<3, error('not enough input'); end
[ntrain dimtrain] = size(X);
Xtrainsum = NaN;

if ~strncmp(kernel,'l',1)
    Xtrainsum = sum(X.*X,2);
    K = repmat(Xtrainsum',ntrain,1);
    K = K + K' - 2*X*X'; % squared distances beetween training smaples, rows of X
end

switch kernel    
    case 'linear'
        K  = X*X'; % linear kernel
        scale = NaN;
    case 'Gaussian'
        scale = parameter*sum(real(sqrt(K(:))))/(ntrain*ntrain); 
        sigma2 = 2*scale^2;
        K = exp(-K/sigma2); % Gaussian
    case 'poly'
        param1 = parameter(1); param2 = parameter(2);
        scale=[param1, param2];
        K = (K + param1) .^ param2; % ploy kernel
    otherwise
        error('Unknown kernel function.');
end

        
        




