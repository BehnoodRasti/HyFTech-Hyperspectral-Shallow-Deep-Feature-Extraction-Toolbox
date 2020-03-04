function [oa, pa, K, CM] = SFE_JPLAY(HSI, Tr, Te, dim, num, sigma, alfa, beta, gamma, rho, maxiter, eta, epsilon, Trees)

[m, n, z] = size(HSI);
HSI2d = hyperConvert2d(HSI);
% data normalization
HSI2d = DataNormlization(HSI2d);

TrainI2d = hyperConvert2d(Tr);
TestI2d = hyperConvert2d(Te);

l_Tr = find(TrainI2d > 0);
l_Te = find(TestI2d > 0);

Samples_Tr = HSI2d(:, l_Tr);
Samples_Te = HSI2d(:, l_Te);
Labels_Tr = TrainI2d(:, l_Tr);
Labels_Te = TestI2d(:, l_Te);

k = 100;
% data clustering to reduce the computional complexity
[NewSamples_Tr, NewLabels_Tr] = ClusterCenter(Samples_Tr, Labels_Tr, k);

Y = GeneLableY(NewLabels_Tr, max(NewLabels_Tr)); % l*N_train: l is the number of class
% Construct adjacency matrix and Laplacian matrix
[G, L] = creatLap(NewSamples_Tr, num, sigma); % Return adjacency matrix G and Laplacian matrix L
% Evenly give the middle subspace dimensions
layer = 5; % Layers: you can tune it accordingly and here we just give an example.
d = generatePath(z, layer, dim); % Generate the dimension sequence for intermediate subspaces
% Run JPLAY to learn projections on train samples
[theta, ~, res] = JPLAY(NewSamples_Tr, Y, G, L, num, d, sigma, alfa, beta, gamma, rho, maxiter, eta, epsilon);
       
fea = [Samples_Tr, Samples_Te];
fea_all=HSI2d;
for i = 1 : length(d)
    fea = theta{1, i} * fea;
    fea_all = theta{1, i} * fea_all;
end

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