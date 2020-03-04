function [ R ] = average_fusion( img,n )
%PCA_FUSION Summary of this function goes here
%   Detailed explanation goes here
no_bands=size(img,3);
for i=1:n
R(:,:,i)= mean(img(:,:,1+floor(no_bands/n)*(i-1):floor(no_bands/n)*i),3);
if (floor(no_bands/n)*i~=no_bands)&(i==n)%当不够等分时，剩下的算作一类。。
    R(:,:,i+1)=mean(img(:,:,0+floor(no_bands/n)*i:no_bands),3);
end    
end
