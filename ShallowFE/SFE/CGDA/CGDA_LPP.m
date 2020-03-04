function M = CGDA_LPP(TrainSample, d, W)

      options.PCARatio =0.9;
      options.ReducedDim=d;
      M= LPP(W, options, TrainSample');
      
end