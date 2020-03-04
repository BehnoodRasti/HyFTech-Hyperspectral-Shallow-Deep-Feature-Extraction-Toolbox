function [W,sigma,lambda,PARAMETERS,Wcand] = LSDR(Y,X,d,insigma,inlambda,b,INPARAM)
%
% Least squares dimension reduction (LSDR)
%
% Usage:
%       [W,sigma,lambda,PARAMETERS,Wcand] = LSDR(Y,X,d,insigma,inlambda,INPARAM)
%
% Input:
%    Y     : dy by n output matrix (iid from density py)
%    X     : dx by n input matrix (iid from density px)
%
%    insigma: array of candidates of Gaussian width, 
%                          one of them is selected by cross validation.
%    inlambda: array of candidates of the regularization parameter, 
%                          one of them is selected by cross validation.
%    b     : number of Gaussian centers
%    INPARAM: parameters which control the behavior of LSDR
%       - output_type: regression or classification 
%                      1: regression
%                      2: classification 
%       - Max_Trial: number of trials with different initialization
%                    (default: 10)
%       - RegEpsilon: the regularization matrix is set to be 
%                     R = K + RegEpsilon*Ib
%
% Output:
%     W: d by dx projection matrix
%     sigma,lambda: Gaussian width and regularization parameter which are
%                   finally chosen
%     PARAMETERS: parameters which were used in LSDR
%        - output_type, Max_Trial, RegEpsilon
%     Wcand: solution W for each initialization condition 
%
% (c) Taiji Suzuki, Department of Mathematical Informatics, The University of Tokyo, Japan. 
%     Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     s-taiji@stat.t.u-tokyo.ac.jp
%     sugi@cs.titech.ac.jp,


Y = Y/std(Y);
X = X/mean(sqrt(var(X,1,2)));

[dor n] = size(X);
[dy ny] = size(Y);
if n~=ny
    error('numbers of samples of x and y must be the same!!!')
end

if nargin < 2
    error('Please specify the dimension of the target space!!!')
end;

if exist('INPARAM','var') && isfield(INPARAM,'output_type') 
    %1:regression, 2:classification, 
    output_type = INPARAM.output_type;
else
    if length(unique(Y)) <= 3
        output_type = 2;
    else
        output_type = 1;
    end;
end;

if exist('INPARAM','var') && isfield(INPARAM,'Max_Trial') 
    Max_Trial = INPARAM.Max_Trial;
else
    Max_Trial = 1;
end;

if exist('INPARAM','var') 
    if isfield(INPARAM,'RegEpsilon') 
        RegEpsilon = INPARAM.RegEpsilon;
    else
        RegEpsilon = 0.01;
    end;
end;

if nargin < 3 || isempty(insigma)
    insigma = 1;%[0.05:0.05:1];
end;

if nargin < 4 || isempty(inlambda)
    inlambda =1;% [10^(-2),10^(-1),10^(0),10^(1),10^(2)];
end;

if nargin < 5 || isempty(b)
    b=200; 
end;
b = min(n,b);
ww = randperm(n);
ind = ww(1:b);

max_iteration = 5; 
epsilon_list= 0.1;%[1 0.1 0.1 0.1 0.1 0.05]; 

Is = Inf;
minIs = Is;

cx = [X(:,ind)];
for mm = 1:dor
    XV{mm} = ones(n,1)*cx(mm,:) - X(mm,:)'*ones(1,size(cx,2));
end;

sig = insigma(1);
lambda = inlambda(1);

Ib = eye(b,b);
IsHistryInit(1) = Inf;

INITIAL_INDEX = [1:Max_Trial];
Init_count = 0;
for Initial =  INITIAL_INDEX
    
    Init_count = Init_count + 1;  
    Isarray = [];
    Whis = [];
    epsilon_count = 0;

    minIs = Inf;

    %random initialization of W%
    Ww = randn(dor,dor); 
    Ww = Ww'*Ww;
    [Ww V D] = svd(Ww);
    W = Ww(1:d,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
    for epsilon=epsilon_list
        b_storeG = 0;
        epsilon_count = epsilon_count + 1;
 
        b_continue = 1;
        for iterate = 1:max_iteration
            %disp(iterate)
            Z = W*X;
            cz = [Z(:,ind)];

            [ZV,ZV2] = ZV_gen(Z,cz,Y,ind);
                        
            if (mod(iterate-1,5) == 0 && epsilon_count == 1) || ((iterate==3 ) && (epsilon_count == 3 )) 
                if ismember(Init_count, [1:10]) 
                    CVinLSDR;
                end;
            end;
            
            if b_storeG
                Gmulti = Gmulti_Store;
                Gmulti2 = Gmulti2_Store;
                Hhat = Hhat_Store;
                GG = GG_Store; 
                hhat = hhat_Store; 
            else
                CalcG;
            end;

            RegMat = GG(ind,:) + Ib*RegEpsilon; 
            
            alpha = pinv(Hhat + lambda*RegMat)*hhat;
            beta  = pinv(Hhat + lambda*RegMat)*(Hhat*alpha);
          
            Is_new = IsCalc(Hhat,hhat,alpha); 

            Is = Is_new;
            if minIs >= Is
                minIs = Is;
            end;

            work = W./(sqrt(sum(W.^2,2))*ones(1,size(W,2)));
            Whis{end+1} = work;
            if  (epsilon_count <= 2 && (iterate == 20)) || b_continue == 0
                b_continue = 1;
                break;
            end;
            
            %calculation of derivative
            Wdir_h = zeros(d,d);
            Wdir_H = zeros(d,d);
            for l = 1:d
                Win = Gmulti2{1};                
                WMAT = ZV{l}.*Gmulti{2};    
                for ll = 1:dor
                    Hin = (WMAT.*(- XV{ll}))'*Gmulti{2}/(n*sig^2);
                    Hin = Hin + Hin';
                    Kin = (-XV{ll}(ind,:).*ZV{l}(ind,:)).*GG(ind,:)/(sig^2);
                    Wdir_H(l,ll) = alpha'*(Win.*Hin)*(-beta + 1.5*alpha) + lambda*(alpha'*Kin*(alpha-beta));
                    Wdir_h(l,ll) = sum((ZV{l}.*(-XV{ll})).*GG,1)*(beta - 2*alpha)/(n*sig^2);
                end;
            end;
            Wdir = -(Wdir_h + Wdir_H);  
            
            %line search or gradient descent
            if epsilon_count <= 1 
                %line search for the first step
                epsilon_line = epsilon;
                Wtmp = W;
                
                WtmpH = Wtmp'*Wdir - Wdir'*Wtmp;  
                WtmpH = WtmpH/max(0,sqrt(trace(WtmpH*WtmpH')));
                
                line_iternum = 20;
                line_iter_array = [0:1:line_iternum]; 
                Is_line = ones(1,line_iternum+1)*(Inf);
                min_Isline = Is;
                Is_line(1) = Is;
                lcount = 1;
                for line_iter =line_iter_array(2:end);
                    lcount = lcount + 1;
                    
                    W = Wtmp*expm(line_iter*epsilon_line*WtmpH);
                    W = W./sqrt((sum(W.^2,2)*ones(1,size(W,2))));
                    
                    DeriveIs;
                    
                    Is_line(lcount) = Is_tmp;
                    min_Isline = min(Is_line(lcount),min_Isline); 
                    if min_Isline == Is_line(lcount)
                        b_storeG = 1;
                        Gmulti_Store = Gmulti;
                        Gmulti2_Store = Gmulti2;
                        Hhat_Store = Hhat;
                        GG_Store = GG; 
                        hhat_Store = hhat;                        
                    end;
                    if min_Isline < Is && Is_line(lcount) > Is_line(lcount-1)
                        break;
                    end;
                end;

                [Is_new minl] = min(Is_line);

                stepsize = line_iter_array(minl);
                W = Wtmp*expm(stepsize*epsilon_line*WtmpH);
                W = W./sqrt((sum(W.^2,2)*ones(1,size(W,2))));
                if stepsize == 0
                    b_continue = 0;
                end;
            else
                %gradient descent
                WtmpH = W'*Wdir - Wdir'*W; 
                b_stop = 1;
                llstep = 0;
                step_mu = 0.1;
                step_beta = 0.3;
                step_alpha = 1;
                Wtmp = W;
                DeriveIs;
                Is_tmp_old = Is_tmp;
                        
                while b_stop %Armijo's rule
                    llstep = llstep + 1;
                    W = Wtmp*expm(step_alpha*(step_beta)^llstep*WtmpH);
                    DeriveIs;
                    if Is_tmp - Is_tmp_old - step_alpha*(step_beta)^llstep*step_mu*trace(WtmpH'*Wtmp'*(-Wdir)) <= 0 || llstep>10
                        break;
                    end;
                end;
                if norm(WtmpH) <  0.00005 || abs(Is_tmp_old - Is_tmp) <= 0.0001 
                    break;
                end;
            end;
        end;
        if norm(WtmpH) <  0.00005
            b_continue = 0; %break;
        end;
    end;
    DeriveCVIs;
    minIs = Is_tmp;
    
    if minIs <= min(IsHistryInit)
        WTrue = Whis{end};
        sigmatrue = sig;
        lambdatrue = lambda;        
    end;
    
    IsHistryInit(Initial) = minIs;
    Wcand{Initial} = Whis{end};
    disp(sprintf('Initial:%d',Initial));
end;
W = WTrue';
sigma = sigmatrue;
lambda = lambdatrue;

PARAMETERS.Max_Trial = Max_Trial;
PARAMETERS.RegEpsilon = RegEpsilon;
PARAMETERS.output_type = output_type;

function [ZV,ZV2] = ZV_gen(Z,cz,Y,ind) 
    [d n]= size(Z);
    b = size(cz,2);
    
    for ii = 1:2
        if ii == 1
            XX = Y;
        else
            XX = Z;
        end;
        cxx = XX(:,ind);
        sqX = sum(XX.^2,1);
        Xc = XX'*cxx;
        sqcx = sum(cxx.^2,1);
        ZV2{ii} = ones(n,1)*sqcx - 2*Xc + sqX'*ones(1,b);
    end;

    for l = 1:d
        ZV{l} = ones(n,1)*cz(l,:) - Z(l,:)'*ones(1,size(cz,2));
    end;
    
function [ZV,ZV2] = ZV_gen2(Z,cz,ind,ZV21) 
    [d n]= size(Z);
    b = size(cz,2);
    
    XX = Z;
    cxx = XX(:,ind);
    sqX = sum(XX.^2,1);
    Xc = XX'*cxx;
    sqcx = sum(cxx.^2,1);
    ZV2 = ones(n,1)*sqcx - 2*Xc + sqX'*ones(1,b);

    for l = 1:d
        ZV{l} = ones(n,1)*cz(l,:) - Z(l,:)'*ones(1,size(cz,2));
    end;

function Is_new = IsCalc(Hhat,hhat,alpha)
    Is_new = alpha'*Hhat*alpha/2 - hhat'*alpha;
