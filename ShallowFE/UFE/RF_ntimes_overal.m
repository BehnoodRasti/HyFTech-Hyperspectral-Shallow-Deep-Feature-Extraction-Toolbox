function [acc_Mean,acc_std,ClMAP]=RF_ntimes_overal(Image,Train,Test,Trees)
% output: 
n=size(Train,3);
for Num_itr=1:n
    T=Train(:,:,Num_itr);
    T1=Test(:,:,Num_itr);
    [nx,ny,nz]=size(T);
    train_label=reshape(T,nx*ny,nz)';
    x1=Image(1:nx,1:ny,:);
    [s1,s2,s3]=size(x1);
    Data=reshape(x1,s1*s2,s3)';
    train_labels=double(train_label(train_label>0));
    X=Data(:,train_label>0);
    [nx1,ny1,nz1]=size(T1);
    test_label=reshape(T1,nx1*ny1,nz1)';
    test_labels=double(test_label(test_label>0));
    model = classRF_train(X',train_labels',Trees);
    res = classRF_predict(Data',model);
    ClMAP=[];
    ClMAP(:,:)=reshape(res,nx,ny);
    predict_labels=double(ClMAP(test_label>0));
    [oa ua pa K]=confusion(test_labels',predict_labels);
    IndMat1(:,Num_itr)=[pa;mean(pa);oa;K];
    IndMat2(:,Num_itr)=[ua;mean(ua)];
    Num_itr;
end
acc_Mean=mean(IndMat1,2);
acc_std=std(IndMat1,[],2);