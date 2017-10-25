function [signal] = dsget(dataset,sensor,activity, volunteer, time_interval)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    signal = dataset{time_interval}{activity, volunteer}(:,sensor);    
end

