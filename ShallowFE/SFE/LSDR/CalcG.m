if exist('Gmulti')
    loopind = [2];
else
    if output_type == 1
        loopind = [1 2];
    else
        loopind = [2];
        Gmulti{1} = 1*(ZV2{1}==0);
        Gmulti2{1} = Gmulti{1}'*Gmulti{1}/n;
    end;
end;
for l = loopind
    Gmulti{l} = exp(-ZV2{l}/(2*sig^2));%exp( - (X(l,:)'*ones(1,size(cx,2))- ones(n,1)*cx(l,:)).^2/(2*sig^2));
end;
GG = Gmulti{1}.*Gmulti{2};


hhat = mean(GG,1)';
hmu = mean(Gmulti{1},1)'.*mean(Gmulti{2},1)';

for l = loopind
    Gmulti2{l} = Gmulti{l}'*Gmulti{l}/n;
end;

Hhat = Gmulti2{1}.*Gmulti2{2};  
