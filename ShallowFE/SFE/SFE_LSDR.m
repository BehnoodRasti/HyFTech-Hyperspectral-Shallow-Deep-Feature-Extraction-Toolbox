function [oa, pa, K, CM] = SFE_LSDR(HSI, Tr, Te, dim, Trees)

[m, n, z] = size(HSI);
HSI2d = hyperConvert2d(HSI);
% data normalization
HSI2d = zscore(HSI2d);

TrainI2d = hyperConvert2d(Tr);
TestI2d = hyperConvert2d(Te);

l_Tr = find(TrainI2d > 0);
l_Te = find(TestI2d > 0);

Samples_Tr = HSI2d(:, l_Tr);
Samples_Te = HSI2d(:, l_Te);
Labels_Tr = TrainI2d(:, l_Tr);
Labels_Te = TestI2d(:, l_Te);

k = 30;
% data clustering to reduce the computional complexity
[NewSamples_Tr, NewLabels_Tr] = ClusterCenter(Samples_Tr, Labels_Tr, k);
M = LSDR(NewLabels_Tr, NewSamples_Tr, dim, [], [], [], 1);

fea = M' * [Samples_Tr, Samples_Te];
fea_all = M' * HSI2d;

train_fea = fea(:, 1 : length(Labels_Tr));
test_fea = fea(:, length(Labels_Tr)+1 : end);

%% Classification with RF
model = classRF_train(train_fea', Labels_Tr', Trees);
classTest = classRF_predict(test_fea', model);
[oa, ua, pa, K, confu]= confusion(Labels_Te', classTest);

%% Generate classificaiton maps 
classAll = classRF_predict(fea_all', model);
CM = hyperConvert3d(classAll,m,n);
end