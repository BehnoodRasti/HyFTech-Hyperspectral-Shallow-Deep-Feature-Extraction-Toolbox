function [oa ua pa K confu]=confusion(true_label,estim_label)
%
% function confusion(true_label,estim_label)
%
% This function compute the confusion matrix and extract the OA, AA
% and the Kappa coefficient.
%
% INPUT
% easy! just read

l=length(true_label);
nb_c=max(true_label);

confu=zeros(nb_c,nb_c);

for i=1:l
  confu(estim_label(i),true_label(i))= confu(estim_label(i),true_label(i))+1;
end

oa=trace(confu)/sum(confu(:)); %overall accuracy
ua=diag(confu)./sum(confu,2);  %class accuracy
pa=diag(confu)./sum(confu,1)';


Po=oa;
Pe=(sum(confu)*sum(confu,2))/(sum(confu(:))^2);

K=(Po-Pe)/(1-Pe);%kappa coefficient


%http://kappa.chez-alice.fr/kappa_intro.htm