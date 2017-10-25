function [feature] = dsgetfeature(dataset,feat_index, sensor, activity,volunteer, time_interval)
%gets a single feature
    feature = dataset{time_interval}{activity, volunteer}{1,sensor}{feat_index};
end

