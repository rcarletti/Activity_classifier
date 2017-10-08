function [dataset] = dsput(dataset,signal,sensor, activity, volunteer)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    dataset{activity, volunteer}(:,sensor) = signal;
end

