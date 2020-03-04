for l = 1:2
    Gmulti{l} = exp(-ZV2{l}/(2*sig^2));
end;
GG = Gmulti{1}.*Gmulti{2};

FOLDNUM = 5;
unitint = floor(n/FOLDNUM);
indvec2 = randperm(n);
centerind = ind;
for ii = 1:FOLDNUM
    foldind{ii} = [[1:(ii-1)*unitint] [ii*unitint+1:n]];
    foldindrest{ii} = [(ii-1)*unitint + 1:ii*unitint];

    centfind = [1:length(ind)];
    centfind_store{ii} = centfind;

    ll = 2;
    hcros{ii} = Gmulti{1}(foldind{ii},centfind)';
    hcros{ii} = hcros{ii}.*Gmulti{ll}(foldind{ii},centfind)';
    hcros{ii} = mean(hcros{ii},2);
    HCROS{ii} = Gmulti{1}(foldind{ii},centfind)'*Gmulti{1}(foldind{ii},centfind)/ length(foldind{ii}); 
    HCROS{ii} = HCROS{ii}.*(Gmulti{ll}(foldind{ii},centfind)'*Gmulti{ll}(foldind{ii},centfind))/length(foldind{ii}); 
    
    hhatcros{ii} = Gmulti{1}(foldindrest{ii},centfind)';
    hhatcros{ii} = hhatcros{ii}.*Gmulti{ll}(foldindrest{ii},centfind)';
    hhatcros{ii} = mean(hhatcros{ii},2);

    Hhatcros{ii} = Gmulti{1}(foldindrest{ii},centfind)'*Gmulti{1}(foldindrest{ii},centfind)/ length(foldindrest{ii}); 
    Hhatcros{ii} = Hhatcros{ii}.*(Gmulti{ll}(foldindrest{ii},centfind)'*Gmulti{ll}(foldindrest{ii},centfind))/length(foldindrest{ii}); 
    
end;

lam = lambda;
val = 0;
for ii = 1:FOLDNUM
    btmp = size(HCROS{ii},1);
    RegMat = GG(centerind(centfind_store{ii}),centfind_store{ii}) + Ib(1:btmp,1:btmp)*RegEpsilon; 
    D = HCROS{ii} + lam*RegMat;
    alcros = D\hcros{ii};
    val = val + IsCalc(Hhatcros{ii},hhatcros{ii},alcros);
end;
val = val/FOLDNUM;
Is_tmp = val;