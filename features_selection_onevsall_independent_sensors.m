%% Perform best feature selection on a 1-vs-all classifier with independent sensors

onevsall_ind = cell(1,4);

%%
for i = 1:4
    [onevsall_ind{i}.best_sensor, onevsall_ind{i}.net] = ...
        feature_selection(i, features_ds);
end

function [sensor, net] = feature_selection(act, features_ds)
    global chosen_features_num;
    global total_features;

    % find (n,k) combinations of features

    C = nchoosek((1:total_features), chosen_features_num);
    nets_num = nchoosek(total_features, chosen_features_num);

    % create target and input vectors for neural network training

    targets = zeros(2, 40);
    targets(2,:) = ones(1, 40);
    targets(1, (1:10) + (act-1)*10) = ones(1,10);
    targets(2, (1:10) + (act-1)*10) = zeros(1,10);

    inputs = cell(1,3);
    for s_id = 1:3
        inputs{s_id} = zeros(4, 40, size(C,1));

        for t_id = 1:size(C,1)
            for a_id = 1:4
                for v_id = 1:10
                    for f_id = 1:chosen_features_num
                        inputs{s_id}(f_id, ((a_id-1) * 10) + v_id, t_id) = ...
                            dsgetfeature(features_ds, C(t_id, f_id), s_id, a_id, v_id);
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
    population = zeros(100, total_features);
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
    options.useParallel = 'true';

    intcon = (1:total_features);
    nonlinearcon = @(x)nonlcon(x);

    % get the best set of features for each sensor

    best_features = cell(1,3);
    for s_id=1:3
        best_features{s_id}.genes = ga(@(x) ...
            fitnessfunction(neural_networks{s_id}, x), ...
            total_features, [], [], [], [], ...
            zeros(1,total_features), ones(1,total_features), nonlinearcon, intcon, options);
        best_features{s_id}.features = genes2feat(best_features{s_id}.genes);
    end

    % compute accuracy for each sensor

    for s_id=1:3
        net = getnetworkbyfeatures(neural_networks{s_id}, best_features{s_id}.features);
        best_features{s_id}.accuracy = 1 - net.conf;
    end

    % choose the best sensor for the classifier

    max_acc = 0;
    best_s_id = 1;

    for s_id=1:3
        if best_features{s_id}.accuracy > max_acc
            max_acc = best_features{s_id}.accuracy;
            best_s_id = s_id;
        end
    end

    sensor.index = best_s_id;
    sensor.features = best_features{best_s_id}.features;
    sensor.accuracy = max_acc;
    
    net = getnetworkbyfeatures(neural_networks{best_s_id}, sensor.features);
end
