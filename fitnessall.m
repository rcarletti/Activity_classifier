function [conf] = fitnessall(features_set,targets,features_ds, type)
%create and train a neural network, using inputs selected by the GA 
%to minimize the confusion

    global chosen_features_num
    global onevsall_all;
    global total_features
    global fcc_all;
    
    features = genes2feat(features_set);
    
    %build the inputs matrix, starting from the features set computed by
    %the GA
    
    inputs = zeros(chosen_features_num, 120);
    
    for a_id =1:4
        for v_id = 1:30
            for f_id = 1:chosen_features_num
                inputs(f_id, ((a_id - 1) * 30 + v_id)) = ...
                    dsgetfeature(features_ds,...
                    mod(features(f_id), total_features) + 1,...     %feature_index
                    ceil(v_id/10),...                               %sensor_id
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
    
    [net,tr] = train(net,inputs,targets);
    
    out = net(inputs);
    conf = confusion(targets, out);
    
    %four class classifier
    if strcmp(type, '4cc')
        fcc_all.net.tr = tr;
        fcc_all.net.accuracy = 1 - conf;
        fcc_all.net.net = net;
    else
        %one versus all classifier
        onevsall_all{str2double(type(1))}.net = net;
        onevsall_all{str2double(type(1))}.tr = tr;
        onevsall_all{str2double(type(1))}.accuracy = 1-conf;
        
    end
 
end

