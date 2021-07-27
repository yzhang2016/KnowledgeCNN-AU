%% compare the names 

load('input.mat') ; 


dset = E ; 
names = [] ; 
flag = 0 ; 
for i = 1: length(dset)
    
    for j = 1 : length(names)
        if strcmp(dset{i},names{j})
            flag = 1 ; 
            break ; 
        end
    end
    if flag == 0
        names = [names;{dset{i}}] ;
    else
        flag = 0 ; 
    end
    
end


