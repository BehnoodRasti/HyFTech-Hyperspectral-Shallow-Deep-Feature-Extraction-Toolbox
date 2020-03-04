function K = centering(K)

% K = centering(K)
%
% Center a kernel matrix
%
% Copyright 2009-2011
% Wenzhi Liao
% wliao@telin.ugent.be, http://telin.ugent.be/~wliao/
% 2 Oct 2010


[nrow nclom] = size(K);
if nrow ~= nclom
    error('input matrix must be symmetric matrax')
end


D = sum(K)/nrow;
E = sum(D)/nrow;
J = ones(nrow,1)*D;
K = K-J-J'+E;
