function [dataset] = dsput(dataset,signal,sensor,activity,volunteer, time_interval)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    dataset{1,time_interval}{activity, volunteer}(:,sensor) = signal;
end

