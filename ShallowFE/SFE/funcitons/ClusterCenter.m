function [NewSamples,NewLabels]=ClusterCenter(SamplesHS,Labels,k)

l=unique(Labels);
opts = statset('Display','final');

NewSamples=[];
NewLabels=[];

for i=1:length(l)
    x=find(Labels==i);
    Data=SamplesHS(:,x);
    if length(x)<=k
        NewSamples=[NewSamples,Data];
        NewLabels=[NewLabels,i*ones(1,length(x))];
    else

    rng(1);
    [idx,C] = kmeans(Data',k,'Start','uniform','distance','cosine','Replicates',1,'MaxIter',1000,'Options',opts);
    NN=[];
    for ii=1:k
        d1=find(idx==ii);
        N=Data(:,d1);
        NN=[NN,mean(N,2)];
    end
     NewSamples=[NewSamples,NN];
     NewLabels=[NewLabels,i*ones(1,k)];
    end
end

end
