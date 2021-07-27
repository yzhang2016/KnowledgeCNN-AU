function [PCC,iCC,MAE,MSE] = OSWMeasure(target,pred)
%% Measure
% PCC : correlation 
% ICC : 
% MAE : mean absolute error 
% MSE: mean square error

% MAE 
MAE = sum(abs(pred - target)) / length(target)  ; 

% MSE 
MSE = sum((pred - target).^2) / length(target) ; 

% ICC
cse = 3;
typ = 'single'; % for computing ICC
iCC = ICC(cse,typ,[target(:),pred(:)]) ; 

% PCC
RR = corrcoef(target(:),pred(:)) ; 
PCC = RR(1,2) ; 


