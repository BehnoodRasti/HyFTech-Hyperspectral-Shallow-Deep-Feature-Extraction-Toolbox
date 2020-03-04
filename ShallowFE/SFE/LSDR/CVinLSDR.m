ll = 1;
if output_type == 1
    Gmulti{ll} = exp(-ZV2{ll}/(2*sig^2));
else 
    Gmulti{ll} = 1*(ZV2{ll}==0);
end;
FOLDNUM = 5;
unitint = floor(n/FOLDNUM);
centerind = ind;
for ii = 1:FOLDNUM
    foldind{ii} = [[1:(ii-1)*unitint] [ii*unitint+1:n]];%[(ii-1)*unitint + 1:ii*unitint];
    foldindrest{ii} = [(ii-1)*unitint + 1:ii*unitint];

    centfind = ismember(centerind,union(foldind{ii},foldind{ii}));    
    centfind_store{ii} = centfind;

    hcros1{ii} = Gmulti{1}(foldind{ii},centfind)';
    HCROS1{ii} = Gmulti{1}(foldind{ii},centfind)'*Gmulti{1}(foldind{ii},centfind)/ length(foldind{ii}); 
    hhatcros1{ii} = Gmulti{1}(foldindrest{ii},centfind)';
    Hhatcros1{ii} = Gmulti{1}(foldindrest{ii},centfind)'*Gmulti{1}(foldindrest{ii},centfind)/ length(foldindrest{ii}); 
end;
    

CV_VALUES = ones(length(insigma),length(inlambda))*Inf;
inlamflag = 1;
isig = 0;
minloocv = Inf;
for sig = insigma
    isig = isig + 1;

    Gmulti{2} = exp(-ZV2{2}/(2*sig^2));
    GG = Gmulti{1}.*Gmulti{2};
        
    indvec2 = randperm(n);
    for ii = 1:FOLDNUM
        foldind{ii} = [[1:(ii-1)*unitint] [ii*unitint+1:n]];%[(ii-1)*unitint + 1:ii*unitint];
        foldindrest{ii} = [(ii-1)*unitint + 1:ii*unitint];
        foldind{ii} = foldind{ii};
        foldindrest{ii} = foldindrest{ii};
        
        centfind = ismember(centerind,union(foldind{ii},foldind{ii}));
        centfind_store{ii} = centfind;
        
        ll = 2;
        hcros{ii} = hcros1{ii}.*Gmulti{ll}(foldind{ii},centfind)';
        hcros{ii} = mean(hcros{ii},2);
        HCROS{ii} = HCROS1{ii}.*(Gmulti{ll}(foldind{ii},centfind)'*Gmulti{ll}(foldind{ii},centfind))/length(foldind{ii}); 

        hhatcros{ii} = hhatcros1{ii}.*Gmulti{ll}(foldindrest{ii},centfind)';
        hhatcros{ii} = mean(hhatcros{ii},2);
        Hhatcros{ii} = Hhatcros1{ii}.*(Gmulti{ll}(foldindrest{ii},centfind)'*Gmulti{ll}(foldindrest{ii},centfind))/length(foldindrest{ii}); 
    end;
    
    inlamfor = [inlamflag:length(inlambda)];
    inlamback = [inlamflag-1:-1:1];
    b_flag = 0;
    ilam = inlamflag;
    b_nowforback = 1;
    lamcount = 0;
    while 1
        lamcount = lamcount + 1;
        if isempty(inlamfor)
            b_nowforback = 2;
        else
            b_nowforback = 1;
        end;

        if ~isempty(inlamfor)
            ilam = inlamfor(1);
            inlamfor = inlamfor(2:end);
            lam = inlambda(ilam);
        elseif ~isempty(inlamback)
            ilam = inlamback(1);
            inlamback = inlamback(2:end);
            lam = inlambda(ilam);
        else
                [aa in] = min(CV_VALUES(isig,:)); 
                inlamflag = in;
            break;
        end;

        val = 0;
        for ii = 1:FOLDNUM
             btmp = size(HCROS{ii},1);
             RegMat = GG(centerind(centfind_store{ii}),centfind_store{ii}) + Ib(1:btmp,1:btmp)*RegEpsilon;
             D = HCROS{ii} + lam*RegMat;
        
            alcros = D\hcros{ii};
            val = val + alcros'*Hhatcros{ii}*alcros/2 - alcros'*hhatcros{ii};        
        end;
        val = val/FOLDNUM;
        CV_VALUES(isig,ilam) =  val;
   
        if min(CV_VALUES(isig,:)) < CV_VALUES(isig,ilam) && lamcount > 1 && isig > 1
            if b_nowforback == 1
                inlamfor = [];
            else
                inlamback = [];
            end;
        end;

        if CV_VALUES(isig,ilam) < minloocv
            minloocv = CV_VALUES(isig,ilam);
        end;
    end;
end;

[mm II] = min(CV_VALUES,[],1);
[m I] = min(mm);
sig = insigma(II(I));
lambda = inlambda(I);
clear Gmulti;

