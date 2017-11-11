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
    nn = struct;
    nn.features = features;
    
    % build the inputs matrix, starting from the features set computed by the GA
    nn.inputs = zeros(chosen_features_num, (40 * time_interval));
    for a_id = 1:4
        for v_id = 1:(10 * time_interval)
            for f_id = 1:chosen_features_num
                nn.inputs(f_id, ((a_id - 1) * 10 * time_interval + v_id)) = ...
                    dsgetfeature(features_ds, ...
                        mod(features(f_id) - 1, total_features) + 1,...     %feature_index
                        ceil(features(f_id) / total_features),...           %sensor_id
                        a_id, v_id, time_interval);
            end
        end
    end

    nn.targets = targets;
    
    % create the NN
    nn.net = patternnet(10);
    nn.net.divideParam.trainRatio = 70/100;
    nn.net.divideParam.valRatio = 15/100;
    nn.net.divideParam.testRatio = 15/100;
    nn.net.trainParam.showWindow = false;

    [nn.net, nn.tr] = train(nn.net,nn.inputs,nn.targets);
    
    nn.results = nn.net(nn.inputs);
    nn.perf = perform(nn.net, nn.targets, nn.results);
    nn.conf = confusion(nn.targets, nn.results);
    nn.accuracy = 1 - nn.conf;
    
    % cache the trained NN
    if strcmp(type, '4cc')
        id = length(fcc_all{time_interval}.nets) + 1;
        fcc_all{time_interval}.nets{id} = nn;
    else
        num = str2double(type(1));
        id = length(onevsall_all{time_interval}{num}.nets) + 1;
        onevsall_all{time_interval}{num}.nets{id} = nn;
    end
 
    conf = nn.conf;
end
