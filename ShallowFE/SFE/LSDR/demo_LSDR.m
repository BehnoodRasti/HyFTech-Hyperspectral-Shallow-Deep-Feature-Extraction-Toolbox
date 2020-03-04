clear all;

dataset=1;
switch dataset
  case 1 % classification
   n=300;
   X1=[randn(2,n).*repmat([1;2],[1 n])+repmat([-3;0],[1 n])];
   X2=[randn(2,n).*repmat([1;2],[1 n])+repmat([ 3;0],[1 n])];
   X=[X1 X2];
   Y=[-ones(1,n) ones(1,n)];
   INPARAM.output_type=2;
   filename='LSDR-classification';
   colormap(jet);
  case 2 % regression
   n=1000;
   X=(rand(2,n)*2-1)*10;
   Y=sin(X(1,:)/10*pi);
   INPARAM.output_type=1;
   filename='LSDR-regression';
   colormap(hsv);
end

INPARAM.Max_Trial = 3;
reduce_dim=1;
W = LSDR(Y,X,reduce_dim,[],[],[],INPARAM);
[~, mapping] = compute_mapping([Y', X'], 'LDA', reduce_dim);
M = mapping.M;
%%%%%%%%%%%%%%%%%%%%%% Displaying original 2D data
figure(1)
clf
hold on

scatter3(X(1,:),X(2,:),Y,100,Y,'filled');
h=plot([-W(1) W(1)]*100,[-W(2) W(2)]*100,'k-','LineWidth',4);
hold on 
plot([-M(1) M(1)]*100,[-M(2) M(2)]*100,'r-','LineWidth',4);
axis equal
axis([-10 10 -10 10])
title('Original 2D data and subspace found by LSDR')

set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 12]);
print('-dpng',filename)
  
