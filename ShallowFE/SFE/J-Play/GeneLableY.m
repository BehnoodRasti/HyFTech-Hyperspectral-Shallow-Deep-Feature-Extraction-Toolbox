function Y=GeneLableY(TrainLabel,Num)

N=length(TrainLabel);
Y=zeros(Num,N);

for i=1:N
    Y(TrainLabel(:,i),i)=1;
end
end