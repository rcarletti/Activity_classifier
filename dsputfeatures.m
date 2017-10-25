function [dataset] = dsputfeatures(dataset,features, sensor, activity,volunteer, time_interval)
%save features for each sensor, for each activity, for each volunteer
    dataset{time_interval}{activity, volunteer}{1,sensor} = features;
end

