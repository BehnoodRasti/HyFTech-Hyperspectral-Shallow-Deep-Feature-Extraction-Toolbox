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