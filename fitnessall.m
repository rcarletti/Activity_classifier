function [conf] = fitnessall(features_set,targets)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    global chosen_features_num
    global total_features
    global best_all
    j = 1;
    features = zeros(1,4);
    for i=1:total_features
        if features_set(i) == 1
            features(j) = i;
            j = j+1;
        end
    end
    
    %build the inputs matrix
    
    inputs = zeros(chosen_features_num, 120);
    
    for a_id =1:4
        for v_id = 1:30
            for f_id = 1:chosen_features_num
                inputs(f_id, ((a_id - 1) * 30 + v_id)) = ...
                    dsgetfeature(features_ds, mod(features(f_id), 11) + 1, v_id,...
                    ceil(v_id/10),...
                    a_id,...
                    mod(v_id, 10) + 1);
            end
        end
    end
    
    net = patternnet(10);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    [net,tr] = train(net,inputs,targets);
    out = net(inputs);
    conf = confusion(targets, out);
    best_all.net = net;
    best_all.accuracy = 1 - conf;
    best_all.tr = tr;
 
end

