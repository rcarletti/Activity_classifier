%% create targets
Z = zeros(1,10);
O = ones(1,10);

targets_onevsall{1} = [O Z Z Z; Z O O O];
targets_onevsall{2} = [Z O Z Z; O Z O O];
targets_onevsall{3} = [Z Z O Z; O O Z O];
targets_onevsall{4} = [Z Z Z O; O O O Z];

%% create and train neural networks for each classifier
neural_networks_vsall = cell(1,4);
best_features_vsall = cell(1,4);
best_sensor_vsall = cell(1,4);

for i = 1:4
    [best_sensor_vsall{i}, best_features_vsall{i}, neural_networks_vsall{i}] = ...
        retrievebestfeatures(i, targets_onevsall{i}, inputs, features_ds);
end

%% local functions

function [sensor, feats, nets] = retrievebestfeatures(class_num, targets, inputs, features_ds)
    global total_features;
    global chosen_features_num;

    %genetic algorithm

    population_size = 100;
    population = zeros(100, total_features * 3);
    rng('shuffle');

    %generate random population (pop_size x total_features array, features set to 1 are
    %the chosen features for that individual

    for i = 1:population_size
        feat_perm = randperm(total_features * 3, chosen_features_num);
        for f_id = 1:chosen_features_num
            population(i, feat_perm(f_id)) = 1;
        end
    end
    
    options = gaoptimset(@ga);
    options.PopulationType = 'doubleVector';
    options.InitialPopulation = population;
    options.useParallel = 'true';

    intcon = (1:33);
    nonlinearcon = @(x) nonlcon(x);
    
    %get the best set of features for each sensor

    feats = cell(1,3);
    for s_id = 1:3
        feats{s_id} = struct;
        feats{s_id}.genes = ga(@(x) fitnessall( ...
            x, targets, features_ds, sprintf('%dvsall', class_num)), ...
            total_features * 3, [], [], [], [], ...
            zeros(1,33), ones(1,33), nonlinearcon, intcon, options);
        feats{s_id}.features = genes2feat(feats{s_id}.genes);
    end
    
    % compute accuracy for each sensor for each classifier
    
    for s_id = 1:3
        net = getnetworkbyfeatures(feats{s_id}.features, s_id, nets);
        feats{s_id}.accuracy = 1 - net.conf;
    end
    
    % choose the best sensor for each activity 
   
    max_acc = 0;
    best_s = 1;
    for s_id = 1:3
        if feats{s_id}.accuracy > max_acc
            max_acc = feats{s_id}.accuracy;
            best_s = s_id;
        end
    end
    
    %convert features from ones to feature numbers

    sensor.index = best_s;
    sensor.features = feats{best_s}.features;
    sensor.accuracy = max_acc;
end
