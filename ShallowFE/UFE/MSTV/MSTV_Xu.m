function fimg=MSTV_Xu(img,n)
%% size of image 
[no_lines, no_rows, no_bands] = size(img);
img2=average_fusion(img,20);
 %% normalization
no_bands=size(img2,3);
fimg=reshape(img2,[no_lines*no_rows no_bands]);
[fimg] = scale_new(fimg);
fimg=reshape(fimg,[no_lines no_rows no_bands]);
%% Structural feature extraction
 fimg1 = tsmooth(fimg,0.003,2);
 fimg2 = tsmooth(fimg,0.02,1);
 fimg3 = tsmooth(fimg,0.01,3);
 f_fimg=cat(3,fimg1,fimg2,fimg3);
 fimg =kpca(f_fimg, 1000,n, 'Gaussian',1000);