function [signal] = dsget(dataset,sensor,activity, volunteer)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    signal = dataset{activity, volunteer}(:,sensor);    
end

