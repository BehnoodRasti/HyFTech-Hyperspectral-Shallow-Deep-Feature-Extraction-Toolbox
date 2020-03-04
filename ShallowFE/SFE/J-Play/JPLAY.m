function [theta,P,res]=JPLAY(X,Y,G,L,k,d,sigma,alfa,beta,gama,rho,maxiter,eta,epsilon)

% Joint & Progressive Learning Strategy (J-PLAY)

% Usage:
%       [theta,P] = JPLAY(X,Y,G,L,k,d,sigma,alfa,beta,gama,maxiter,eta)

%% Input:
%       X      -Input data: d*N (d: dimension, N: the number of sample)
%       Y      -Label matrix: c*N (c: the number of class), 
%                             e.g. if class==1 then y=[1,0,0,...,0]'
%                                  if class==2 then y=[0,1,0,...,0]'
%       G      -Adjacency matrix: N*N
%       L      -Laplacian matrix: N*N
%       k      -The number of nearest neighbor
%       d      -Reduced dimension
%     sigma    -Gaussian kernel parameter used in constructing graph
%     alfa     -Reconstruction loss parameter (default: 1)
%     beta     -Manifold regularization parameter (default: 0.1)
%     gama     -Ridge regression parameter  (default: 0.1)
%    maxiter   -Maximum number of iterations (default: 1000)
%     eta      -Eta in AutoRULe (default: 0.1)

%% Output:
%     theta    -A series of coupled-projection matrices
%       P      -Property-labeled projection matrix

% (c) Danfeng Hong, Remote Sensing Technology Institute (IMF), German Aerospace Center (DLR), Germany.
%                   Singnal Processing in Earth Oberservation (SiPEO), Technical University of Munich (TUM), Germany. 
%     Naoto Yokoya, Remote Sensing Technology Institute (IMF), German Aerospace Center (DLR), Germany.
%                   Singnal Processing in Earth Oberservation (SiPEO), Technical University of Munich (TUM), Germany. 
%                   Research Center for Advanced Science and Technology, The Univerisity of Tokyo, Japan.
%     danfeng.hong@dlr.de
%     naoto.yokoya@dlr.de; yokoya@sal.rcast.u-tokyo.ac.jp 

if nargin < 2
    error('Please specify label information!')
end;

if nargin < 3
    error('Please construct adjacency matrix!')
end;

if nargin < 4
    error('Please construct Laplacian matrix!')
end;

if nargin < 5
    error('Please specify the number of nearest neighbor!')
end;

if nargin < 6
    error('Please specify subspace dimensions!')
end;

if nargin < 7
    error('Please specify Gaussian kernel parameter used in constructing graph!')
end;

if nargin < 8 || isempty(alfa)
    alfa=1;
end;

if nargin < 9 || isempty(beta)
    beta=0.1;
end;

if nargin < 10 || isempty(gama)
    gama=0.1;
end;

if nargin < 11 || isempty(maxiter)
    maxiter=1000;
end;

if nargin < 12 || isempty(eta)
    eta=0.1;
end;

%% Parameters Setting
% epsilon = 1e-3; % Tolerance error 1e-3Houston2018
iter=1; 
stop = false;
res=zeros(1,maxiter); % Residuals
Num=length(d); % The number of layers

%% JPL: Initialization-step
theta=cell(1,1);
X0=X;
for i=1:Num
       theta0=DR_LPP(X0,k,d(i),sigma,G);
        theta_init=AutoRULe(theta0'*X0,X0,L,eta,theta0',maxiter);
        X0=theta_init*X0;
        theta{1,i}=theta_init;
end

%% JPL: Fine-tuning parameters
while ~stop && iter < maxiter+1
    
    %% Solve W
    D=X;
    for i=1:Num
        D=theta{1,i}*D;  
    end
    P=(alfa*(Y*D'))/(alfa*(D*D')+gama*eye(size(D*D')));  
    
    %% Solve the group of theta    
    for j=1:Num
        
        %give W
        Pi=P;
        if j<Num
            for z=Num:-1:j+1
                Pi=Pi*theta{1,z};
            end
        end
        
        %give H
        H=X;
        for m=1:j
            H=theta{1,m}*H;
        end
        
        %give X
        Xi=X;
%         XWi=XW;
        if j>1
           for n=1:j-1
               Xi=theta{1,n}*Xi;
%                XWi=theta{1,n}*XWi;
           end
        end
        
        theta{1,j}=Theta_ADMM(Y,Pi,H,Xi,L,alfa,beta,theta{1,j},rho,maxiter);   
    end

    %% Compute the vaules of objection function 
     ErrorTerm=X;
%      ErrorTerm1=XW;
     ManifoldTerm=0;
%      ManifoldTerm1=0;
     ReconstructionTerm=0;
     for r=1:Num
         ReconstructionTerm=ReconstructionTerm+(norm(ErrorTerm-theta{1,r}'*theta{1,r}*ErrorTerm,'fro')^2);
         ErrorTerm=theta{1,r}*ErrorTerm;
%          ErrorTerm1=theta{1,r}*ErrorTerm1;
         ManifoldTerm=ManifoldTerm+trace(ErrorTerm*L*ErrorTerm');
%          ManifoldTerm1=ManifoldTerm1+trace(ErrorTerm1*L1*ErrorTerm1');
     end
     res(1,iter)=0.5*alfa*(norm(Y-P*ErrorTerm,'fro')^2)+0.5*beta*(ManifoldTerm)+0.5*gama*(norm(P,'fro')^2)+0.5*(ReconstructionTerm);
    
    %% Check the convergence condition
    if iter>1
       r_Obj=abs(res(1,iter)-res(1,iter-1))/res(1,iter-1);
       if r_Obj<epsilon
            stop = true;
            fprintf(' i = %f,res_Obj= %f\n',iter,r_Obj);
            break;
       end

       if mod(iter,10) == 1
           fprintf(' i = %f,res_Obj= %f\n',iter,r_Obj);
       end
    end

    iter=iter+1;
end
end