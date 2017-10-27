function [conf] = fitnessall(features_set,targets,features_ds, type, time_interval)
%create and train a neural network, using inputs selected by the GA 
%to minimize the confusion

    global chosen_features_num
    global onevsall_all;
    global total_features
    global fcc_all;
    
    features = genes2feat(features_set);
    
    %build the inputs matrix, starting from the features set computed by
    %the GA
    
    inputs = zeros(chosen_features_num, (120 * time_interval));
    
    for a_id =1:4
        for v_id = 1:(30 * time_interval)
            for f_id = 1:chosen_features_num
                inputs(f_id, ((a_id - 1) * 30  * time_interval + v_id)) = ...
                    dsgetfeature(features_ds,...
                    mod(features(f_id), total_features) + 1,...     %feature_index
                    ceil(v_id/(10 * time_interval)),...             %sensor_id
                    a_id,...
                    mod(v_id, (10 * time_interval)) + 1,...
                    time_interval);
            end
        end
    end
    
    %create the nn
    net = patternnet(10);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    net.trainParam.showWindow = false;
    
    [net,tr] = train(net,inputs,targets);
    
    out = net(inputs);
    conf = confusion(targets, out);
    
    %four class classifier
    if strcmp(type, '4cc')
        fcc_all{time_interval}.net.tr = tr;
        fcc_all{time_interval}.net.accuracy = 1 - conf;
        fcc_all{time_interval}.net.net = net;
    else
        %one versus all classifier
        onevsall_all{time_interval}{str2double(type(1))}.net.tr = tr;
        onevsall_all{time_interval}{str2double(type(1))}.net.accuracy = 1-conf;
        onevsall_all{time_interval}{str2double(type(1))}.net.net = net;
    end
 
end

