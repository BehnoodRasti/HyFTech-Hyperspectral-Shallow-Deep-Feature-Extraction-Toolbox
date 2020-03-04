function [ OA,AA,kappa,CA ] = MSTV( img,GroundT )
%% size of image 
[no_lines, no_rows, no_bands] = size(img);
GroundT=GroundT';
OA=[];AA=[];kappa=[];w=10;CA=[];
load (['.\training_indexes\in_1.mat'])
img2=average_fusion(img,20);
 %% normalization
no_bands=size(img2,3);
fimg=reshape(img2,[no_lines*no_rows no_bands]);
[fimg] = scale_new(fimg);
fimg=reshape(fimg,[no_lines no_rows no_bands]);
%% feature extraction
 fimg1 = tsmooth(fimg,0.003,2);
 fimg2 = tsmooth(fimg,0.02,1);
 fimg3 = tsmooth(fimg,0.01,3);
 f_fimg=cat(3,fimg1,fimg2,fimg3);
 fimg =kpca(f_fimg, 1000,30, 'Gaussian',1000);

 %% SVM classification
fimg = ToVector(fimg);
fimg = fimg';
fimg=double(fimg);
for i=1:w
indexes=dph(:,i);
%%% traing and test samples
train_SL = GroundT(:,indexes);
train_samples = fimg(:,train_SL(1,:))';
train_labels= train_SL(2,:)';
%
test_SL = GroundT;
test_SL(:,indexes) = [];
test_samples = fimg(:,test_SL(1,:))';
test_labels = test_SL(2,:)';

[train_samples,M,m] = scale_func(train_samples);
[fimg11 ] = scale_func(fimg',M,m);
% Selecting the paramter for SVM
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);
% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_labels,train_samples,parameter);
% Testing
Result = svmpredict(ones(no_lines*no_rows,1),fimg11,model); 
% Evaluation
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);
[OA_i,AA_i,kappa_i,CA_i]=confusion(GroudTest,ResultTest);
Result = reshape(Result,no_lines,no_rows);
VClassMap=label2colord(Result,'india');
figure,imshow(VClassMap);
OA=[OA OA_i];
AA=[AA AA_i];
kappa=[kappa kappa_i];
CA=[CA CA_i];
end


end

