function [dataset] = dsputfeatures(dataset,features, sensor, activity,volunteer)
%save features for each sensor, for each activity, for each volunteer
    dataset{activity, volunteer}{1,sensor} = features;
end

