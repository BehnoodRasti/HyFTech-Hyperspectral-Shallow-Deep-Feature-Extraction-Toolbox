
clear
clc

load('Indian_pines.mat')
x1 = double(indian_pines);
T = imread('IndianTR123_temp123.tif');
T1 = imread('IndianTE123_temp123.tif');
[nx,ny,nz]=size(T);
XXX =reshape(T,nx*ny,nz)';
train_label=reshape(T,nx*ny,nz)';

[s1,s2,s3]=size(x1);
Data=reshape(x1,s1*s2,s3)';

 for i=1:s3
     Data(i,:)=double(mat2gray(Data(i,:)));
 end

train_labels=double(train_label(train_label>0));
X=Data(:,train_label>0);


[nx1,ny1,nz1]=size(T1); 
test_label=reshape(T1,nx1*ny1,nz1)';
test_labels=double(test_label(test_label>0));
X2=Data(:,test_label>0);

x=X2';y=test_labels';
tic
model = classRF_train(X',train_labels',500);

%res(:,ii) = classRF_predict(Data',model);
res = classRF_predict(Data',model);
%---------------------------

t = classRF_predict(x,model);
Time = toc
[sortedlabels,sidx]=sort(test_labels);

Nc=length(unique(test_labels));

for i=1:Nc
    cl=find(sortedlabels==i);
    s=cl(1);e=cl(length(cl));
    sv=t(sidx);
    pcl=sv(s:e);
    for j=1:Nc
        C(j,i)=length(find(pcl==j));
    end
    Cacc(i)=length(find(pcl==i))/length(pcl)*100;
    clear cl pcl;
end
N=sum(sum(C));
sumC=sum(C);
sumR=sum(C');
S=0;
for i=1:Nc
    acc(i)=C(i,i)/sumC(i)*100;
    S=S+sumC(i)*sumR(i);
end
trace(C);
meanacc=mean(acc)
OA=trace(C)/N*100
Po=trace(C)/N;
Pe=S/N^2;
kappa=(Po-Pe)/(1-Pe)*100
%------------------------------
res2=[];
 figure
 res2(:,:)=reshape(res,nx,ny);
 imagesc(res2);
 axis off
axis equal
 title(['Classification using RF,meanacc=',num2str(meanacc), ',OA=',num2str(OA), ',kappa=',num2str(kappa)])
