function [conf] = fitnessfunction(feature_set, sensor_num)
%retrieves the nn trained with the features specified in the feature set
%and computes the confusion associated to it.
    global total_features
    j = 1;
    features = zeros(1,4);
    for i=1:total_features
        if feature_set(i) == 1
            features(j) = i;
            j = j+1;
        end
    end

    nn = getnetworkbyfeatures(features, sensor_num);
    conf = 1-nn.conf;
end
