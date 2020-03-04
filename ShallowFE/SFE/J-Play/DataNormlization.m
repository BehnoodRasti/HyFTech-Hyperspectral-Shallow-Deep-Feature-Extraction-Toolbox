function Normlized_Data=DataNormlization(Data)

    [D,N]=size(Data);
    Normlized_Data=zeros(D,N);
    
    for i=1:N
         Normlized_Data(:,i)=Data(:,i)/(max(Data(:,i)));
    end

end