%% Perform best feature selection on a 4-class classifier with independent sensors

%create a structure for each time interval
fcc_ind = cell(1,4);
for time_interval = [1,2,4]
    fcc_ind{time_interval} = struct;
    [fcc_ind{time_interval}.results, fcc_ind{time_interval}.best_sensor, fcc_ind{time_interval}.net] = ...
        feature_selection(features_ds, time_interval);
end

function [results, sensor, net] = feature_selection(features_ds, time_interval)
    global chosen_features_num;
    global total_features;

    % find (n,k) combinations of features

    C = nchoosek((1:total_features), chosen_features_num);
    nets_num = nchoosek(total_features, chosen_features_num);
    
    % create target and input vectors for neural network training

    Z = zeros(1,(10 * time_interval));
    O = ones(1,(10 * time_interval));

    targets = [O Z Z Z ; Z O Z Z ; Z Z O Z ; Z Z Z O ];

    inputs = cell(1,3);
    for s_id = 1:3
        inputs{s_id} = zeros(4, 40 * time_interval, size(C,1));

        for t_id = 1:size(C,1)
            for a_id = 1:4
                for v_id = 1:(10 * time_interval)
                    for f_id = 1:chosen_features_num
                        inputs{s_id}(f_id, ((a_id - 1) * 10 * time_interval) + v_id, t_id) = ...
                            dsgetfeature(features_ds, C(t_id, f_id), s_id, a_id, v_id, time_interval);
                    end
                end
            end
        end
    end

    % create (n,k) neural networks for each sensor and train them

    neural_networks = cell(1,3);
    neural_networks{1} = createandtrainnn(1, inputs, targets, nets_num, C);
    neural_networks{2} = createandtrainnn(2, inputs, targets, nets_num, C);
    neural_networks{3} = createandtrainnn(3, inputs, targets, nets_num, C);

    %generate random population (pop_size x total_features array, features set to 1 are
    %the chosen features for that individual

    population_size = 100;
    population = zeros(population_size, total_features);
    rng('shuffle');

    for i=1:population_size
        feat_perm = randperm(total_features, chosen_features_num);
        for f_id =1:chosen_features_num
            population(i,feat_perm(f_id)) = 1;
        end
    end

    options = gaoptimset(@ga);
    options.PopulationType = 'doubleVector';
    options.InitialPopulation = population;
    options.Generations = 100;
    options.useParallel = 'true';

    intcon = (1:total_features);
    nonlinearcon = @(x)nonlcon(x);

    % get the best set of features for each sensor

    results = cell(1,3);
    f_tot = total_features; % for performance reasons

    parfor s_id=1:3
        results{s_id}.genes = ga(@(x) ...
            fitnessfunction(neural_networks{s_id}, x), ...
            f_tot, [], [], [], [], zeros(1, f_tot), ones(1, f_tot), ...
            nonlinearcon, intcon, options);
        results{s_id}.features = genes2feat(results{s_id}.genes);
    end

    % compute accuracy for each sensor

    for s_id=1:3
        results{s_id}.net = getnetworkbyfeatures(neural_networks{s_id}, results{s_id}.features);
        results{s_id}.accuracy = 1 - results{s_id}.net.conf;
    end

    % choose the best sensor for the classifier
 
    max_acc = 0;
    best_s_id = 1;

    for s_id=1:3
        if results{s_id}.accuracy > max_acc
            max_acc = results{s_id}.accuracy;
            best_s_id = s_id;
        end
    end

    sensor.index = best_s_id;
    sensor.features = results{best_s_id}.features;
    sensor.accuracy = max_acc;
    
    net = getnetworkbyfeatures(neural_networks{best_s_id}, sensor.features);
end
