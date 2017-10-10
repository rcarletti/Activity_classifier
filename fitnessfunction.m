function [conf] = fitnessfunction(feature_set,sensor_num, inputs, targets)
%retrieves the nn trained with the features specified in the feature set
%and computes the confusion associated to it.
    global total_features
    j = 1;
    %features(1,4) = [];
    for i=1:total_features
        if feature_set(i) == 1
            features(j) = i;
            j = j+1;
            if j == 5
                break;
            end
        end
    end
    
    disp(features)
      
    nn = getnetworkbyfeatures(features, sensor_num);
    results = nn(inputs{sensor_num}(:,:,1));
    conf = confusion(targets(:,:,1), results);
   
end

