function [conf] = fitnessall(features_set, targets, features_ds, type, time_interval)
%create and train a neural network, using inputs selected by the GA 
%to minimize the confusion

    global chosen_features_num;
    global total_features;
    global onevsall_all;
    global fcc_all;
    
    features = genes2feat(features_set);
    
    % first try to find if a NN already exists for these features
    % if it has already been computed, use the cached value
    
    if strcmp(type, '4cc')
        nets = fcc_all{time_interval}.nets;
    else
        nets = onevsall_all{time_interval}{str2double(type(1))}.nets;
    end
    
    for i = 1:length(nets)
        if isequal(nets{i}.features, features)
            conf = 1 - nets{i}.accuracy;
            return
        end
    end
    
    % otherwise, create and train the NN and add it to the NN cache
    
    % build the inputs matrix, starting from the features set computed by the GA
    inputs = zeros(chosen_features_num, (120 * time_interval));
    for a_id = 1:4
        for v_id = 1:(10 * 3 * time_interval)
            for f_id = 1:chosen_features_num
                inputs(f_id, ((a_id - 1) * 30  * time_interval + v_id)) = ...
                    dsgetfeature(features_ds, ...
                        mod(features(f_id) - 1, total_features) + 1,...     %feature_index
                        ceil(v_id / (10 * time_interval)),...               %sensor_id
                        a_id,...
                        mod(v_id - 1, (10 * time_interval)) + 1,...
                        time_interval);
            end
        end
    end
    
    % create the NN
    net = patternnet(10);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    net.trainParam.showWindow = false;
    
    [net,tr] = train(net,inputs,targets);
    
    out = net(inputs);
    conf = confusion(targets, out);
    
    % cache the trained NN
    if strcmp(type, '4cc')
        id = length(fcc_all{time_interval}.nets) + 1;

        fcc_all{time_interval}.nets{id} = struct;
        fcc_all{time_interval}.nets{id}.tr = tr;
        fcc_all{time_interval}.nets{id}.accuracy = 1 - conf;
        fcc_all{time_interval}.nets{id}.net = net;
        fcc_all{time_interval}.nets{id}.features = features;
    else
        num = str2double(type(1));
        id = length(onevsall_all{time_interval}{num}.nets) + 1;

        onevsall_all{time_interval}{num}.nets{id} = struct;
        onevsall_all{time_interval}{num}.nets{id}.tr = tr;
        onevsall_all{time_interval}{num}.nets{id}.accuracy = 1 - conf;
        onevsall_all{time_interval}{num}.nets{id}.net = net;
        onevsall_all{time_interval}{num}.nets{id}.features = features;
    end
 
end
