function d=generatePath(BandNum,Portion,ReducedDim)

RestBand=BandNum-ReducedDim;
n=RestBand/Portion;
tn=n/10;
nu=single(tn-fix(tn));

if nu>=0.3&nu<=0.7
    fn=10*(fix(tn)+0.5);
else
    fn=10*round(tn);
end

d=zeros(1,Portion);

for i=1:Portion
    if i==1
        d(1,i)=ReducedDim;
    else
        d(1,i)=d(1,i-1)+fn;
    end
end
d=fliplr(d);
end