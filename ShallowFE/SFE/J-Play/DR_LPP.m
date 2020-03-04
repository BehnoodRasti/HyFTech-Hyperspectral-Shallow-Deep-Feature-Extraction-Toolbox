function [M,W]=DR_LPP(TrainSample,k,d,sigma,W)
    
      fea=TrainSample';
      options = [];
      options.NeighborMode = 'KNN';
      options.k = k;
      options.WeightMode = 'HeatKernel';
      options.t = sigma;

    if ~exist('W','var')
         W = constructW(fea,options);
    end

      options.PCARatio =1;
      options.ReducedDim=d;
      M= LPP(W, options, fea);
end