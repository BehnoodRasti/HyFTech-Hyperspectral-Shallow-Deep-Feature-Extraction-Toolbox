function [FE]=OTVCA_V3(Y,r_max,lambda,tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code implements a new-version of OTVCA which does not include
% whitening stage.
% OTVCA was illustrated in the following Paper. 
%
% B. Rasti, M. O. Ulfarsson, and J. R. Sveinsson,” Hyperspectral Feature Extraction Using Total Variation Component Analysis”, 
% IEEE Trans. Geoscience and Remote Sensing, 54 (12), 6976-6985.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function can be used as either of the following cases;
% 
% [FE]=OTVCA_V3(Y,r_max,lambda,tol);
% [FE]=OTVCA_V3(Y,r_max,lambda);
% [FE]=OTVCA_V3(Y,r_max);
%
% Input
% Y: Hyperspectral image (3D matrix).
% r_max: Number of features to extract, this could be selected equal to the
% number of classes of interests in the scene
% tol: Number of iterations; 200 iterations are default value
% lambda: Tuning parameter; Default is 0.01 for normalized HSI 
%
% output
% FE: Hyperspectral features extracted (3D matrix)
%
% Example
% See Demo
%
% (c) 2016 Written by Behnood Rasti
% email: behood.rasti@gmail.com
if nargin<4
    tol=200;
end
if nargin<3
    lambda=0.01;
end
[nr1,nc1,p1]=size(Y);
RY=reshape(Y,nr1*nc1,p1);
%Y=reshape(RY,nr1,nc1,p1); % mean value recustion
%RY=RY-mean(RY);
m = min(Y(:));
M = max(Y(:));
NRY=(RY-m)/(M-m);
[~,~,V1] = svd(NRY,'econ');
V=V1(:,1:r_max);
FE=zeros(nr1,nc1,r_max);
for fi=1:tol
    C1=NRY*V(:,1:r_max);
    PC=reshape(C1,nr1,nc1,r_max);
    for j = 1:r_max
        FE(:,:,j)=splitBregmanROF(PC(1:nr1,1:nc1,j),1/lambda,.1);
    end
    fprintf('%d\t',fi)
    RFE=reshape(FE,nr1*nc1,r_max);
    M=NRY'*RFE;
    [C,~,G] = svd(M,'econ');
    V=(C*G');
end