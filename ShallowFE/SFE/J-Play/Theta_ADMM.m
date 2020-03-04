function theta=Theta_ADMM(Y,W,H,X,L,alfa,beta,theta,rho,maxiter)

%% 
%  Solve the JPLAY's subproblem: theta, using ADMM

%% Initializing Setting
epsilon = 1e-6;
iter=0;

G=zeros(size(theta));
Q=zeros(size(theta*X));
P=zeros(size(theta*X));
% M=zeros(size(theta*X));
lamda1=zeros(size(theta*X));
lamda2=zeros(size(theta));
lamda3=zeros(size(theta*X));
lamda4=zeros(size(theta*X));
% lamda5=zeros(size(theta*X));

stop = false;
mu=1e-3;
% rho=2;%1.4:Houston2018 2:Houston2013
mu_bar=1e+6;

GL=(X*L*X');%Graph Laplacian 

  while ~stop && iter < maxiter+1
    
      iter=iter+1;
      %solve theta
      theta=(mu*(H*X')+lamda1*X'+mu*G+lamda2+mu*(Q*X')+lamda3*X'+mu*(P*X')+lamda4*X')/(3*mu*(X*X')+beta*GL+mu*eye(size(X*X')));   
      %solve H
      H=(alfa*(W'*W)+(G*G')+mu*eye(size(W'*W)))\(alfa*(W'*Y)+(G*X)+mu*(theta*X)-lamda1);
      %solve G
      G=((H*H')+mu*eye(size(H*H')))\(mu*theta-lamda2+(H*X'));      
      %solve Q
      Q=max(theta*X-(lamda3/mu),0);  
      %solve P
      MidV=theta*X-lamda4/mu;     
      for i=1:size(MidV,2)
          if norm(MidV(:,i))<=1
             P(:,i)=MidV(:,i);
          else
             P(:,i)=MidV(:,i)/norm(MidV(:,i));
          end
      end
%       M=max(abs(theta*(X-XW)-lamda5/mu)-(beta*0.01/mu),0).*sign(theta*(X-XW)-lamda5/mu); 
     %update Lagrange multipliers  
     lamda1=lamda1+mu*(H-theta*X);
     lamda2=lamda2+mu*(G-theta);
     lamda3=lamda3+mu*(Q-theta*X);
     lamda4=lamda4+mu*(P-theta*X);
%      lamda5=lamda5+mu*(M-theta*(X-XW));
     %update penalty parameter
     mu=min(mu*rho,mu_bar);
     %computer errors
     r_H=norm(H-theta*X,'fro');
     r_G=norm(G-theta,'fro');
     r_Q=norm(Q-theta*X,'fro');
     r_P=norm(P-theta*X,'fro');
%      r_M=norm(M-theta*(X-XW),'fro');
     %check the convergence conditions
     if r_H<epsilon&&r_G<epsilon&&r_Q<epsilon&&r_P<epsilon%&&r_M<epsilon
         stop = true;
         break;
     end
  end
end