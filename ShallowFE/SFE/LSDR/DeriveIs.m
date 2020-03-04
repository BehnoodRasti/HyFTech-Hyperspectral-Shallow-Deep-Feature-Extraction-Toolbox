Z = W*X;cz = [Z(:,ind)];

if exist('ZV2')
    [ZV,ZV2{2}] = ZV_gen2(Z,cz,ind,ZV2{1}); 
else
    [ZV,ZV2] = ZV_gen(Z,cz,Y,ind);
end;

CalcG;

RegMat = GG(ind,:) + Ib*RegEpsilon; 

alpha = (Hhat + lambda*RegMat)\hhat;
Is_tmp = IsCalc(Hhat,hhat,alpha); 