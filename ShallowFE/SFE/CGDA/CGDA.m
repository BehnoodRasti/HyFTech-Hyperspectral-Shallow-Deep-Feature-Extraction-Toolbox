function M = CGDA(X, Y, d, lamda)

% X:  training samples
% Y:  training labels
% d:  reduced dimension d * N

[~, N] = size(X);

%% compute the block-wise graph (W) based on the labels

W = zeros(N, N);
num_labels = length(unique(Y));
L = [];

for i = 1 : num_labels   
    ind = (find(Y == i));
    l = length(ind);
    L = [L, l];
    x = X(:, ind);
    w = (x' * x + lamda * eye(size(x' * x))) \ (x' * x);
    if i == 1
        W(1 : l, 1 : l) = w;
    else 
        W(sum(L(1 : i-1)) +1 : sum(L(1 : i)), sum(L(1 : i-1)) +1 : sum(L(1 : i))) = w;
    end
end

W = W - diag(diag(W));
W = max(W, W');

%% embedding
M = CGDA_LPP(X, d, W); % M:  learned mapping

end