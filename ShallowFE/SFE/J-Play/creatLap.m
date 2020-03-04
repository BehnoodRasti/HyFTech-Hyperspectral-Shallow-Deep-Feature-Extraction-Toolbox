function [W,L]=creatLap(X,k,sigma) 
     
      X=X';
      options = [];
      options.NeighborMode = 'KNN';
      options.k = k;
      options.WeightMode = 'HeatKernel';
      options.t = sigma;
      W = constructW(X,options);
      L=diag(sum(W,2))-W;
      L=full(L);
      
end