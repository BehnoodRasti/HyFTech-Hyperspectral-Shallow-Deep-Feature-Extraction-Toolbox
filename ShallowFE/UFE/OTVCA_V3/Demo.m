% Demo: OTVCA Hyperspectral Feature Extraction
%
load Indian_site
FN=16; % Number of features to extract
[FE]=OTVCA_V3(R,FN);
%% Extrected Features
figure(1)
subplot(1,3,1),imagesc(FE(:,:,1));colormap(gray);axis image;axis off;title('Feature 1');
subplot(1,3,2),imagesc(FE(:,:,2));colormap(gray);axis image;axis off;title('Feature 2');
subplot(1,3,3),imagesc(FE(:,:,3));colormap(gray);axis image;axis off;title('Feature 3');
