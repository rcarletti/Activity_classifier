function [feature] = dsgetfeature(dataset,feat_index, sensor, activity,volunteer)
%gets a single feature
    feature = dataset{activity, volunteer}{1,sensor}{feat_index}
end

