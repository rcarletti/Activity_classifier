function [conf] = fitnessall(features_set,targets,features_ds, type)
%create and train a neural network, using inputs selected by the GA 
%to minimize the confusion

    global chosen_features_num
    global total_features
    global best_all_4cc
    global best_onevsall_all
    j = 1;
    features = zeros(1,4);
    for i=1:total_features * 3
        if features_set(i) == 1
            features(j) = i;
            j = j+1;
        end
    end
    
    %build the inputs matrix, starting from the features set computed by
    %the GA
    
    inputs = zeros(chosen_features_num, 120);
    
    for a_id =1:4
        for v_id = 1:30
            for f_id = 1:chosen_features_num
                inputs(f_id, ((a_id - 1) * 30 + v_id)) = ...
                    dsgetfeature(features_ds,...
                    mod(features(f_id), 11) + 1,...     %feature_index
                    ceil(v_id/10),...                   %sensor_id
                    a_id,...
                    mod(v_id, 10) + 1);
            end
        end
    end
    
    %create the nn
    net = patternnet(10);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    %train the nn five times
    for i = 1:5
        [net,tr] = train(net,inputs,targets);
    end
    
    out = net(inputs);
    conf = confusion(targets, out);
    
    %four class classifier
    if strcmp(type, '4cc')
        best_all_4cc.net = net;
        best_all_4cc.accuracy = 1 - conf;
        best_all_4cc.tr = tr;
    else
        %one versus all classifier
        best_onevsall_all{str2num(type(1))}.net = net;
        best_onevsall_all{str2num(type(1))}.tr = tr;
        best_onevsall_all{str2num(type(1))}.accuracy = 1-conf;
        
    end
 
end

