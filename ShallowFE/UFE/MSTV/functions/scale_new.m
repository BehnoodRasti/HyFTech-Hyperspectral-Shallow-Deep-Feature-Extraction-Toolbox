 function [data M m] =scale_new(data,M,m)
%
% function data =rescale(data)
%
% This function rescale the input data between 0 and 1
%
% INPUT
%
% data: the data to rescale
% max: the maximum value of the ouput data
% min: the minimum value of the output data
% 
% OUTPUT
%
% data: the rescaled data
[Nb_s Nb_b]=size(data);
if nargin==1
    M=max(data,[],1);
    m=min(data,[],1);
end

data = (data-repmat(m,Nb_s,1))./(repmat(M-m,Nb_s,1));

