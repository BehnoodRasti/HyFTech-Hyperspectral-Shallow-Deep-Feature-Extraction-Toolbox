
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
    



