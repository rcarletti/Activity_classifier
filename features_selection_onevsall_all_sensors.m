%% Perform best feature selection on a 1-vs-all classifier with all sensors
global onevsall_all;

onevsall_all = cell(1,4);
for time_interval = [1,2,4]
    onevsall_all{time_interval} = cell(1,4);

    for act = 1:4
        onevsall_all{time_interval}{act} = struct;
        onevsall_all{time_interval}{act}.nets = cell(1,0);
        feature_selection(act, features_ds, time_interval);
        
    end
end

function feature_selection(act, features_ds, time_interval)
    global total_features;
    global chosen_features_num;
    global onevsall_all;
    
    % generate targets for class act
   
    targets = zeros(2, 40 * 3 * time_interval);
    targets(2,:) = ones(1, 40 * 3 * time_interval);

    targets(1, (1:(30 * time_interval)) + (act-1) * 30 * time_interval) = ones(1, 30 * time_interval);
    targets(2, (1:(30 * time_interval)) + (act-1) * 30 * time_interval) = zeros(1, 30 * time_interval);

    % set up the GA, this time we consider 23 features for each sensor

    population_size = 100;
    population = zeros(population_size, total_features * 3);

    % generate random population (pop_size x total_features) array
    % features set to 1 are the chosen features for that individual

    for i = 1:population_size
        feat_perm = randperm(total_features * 3, chosen_features_num);
        for f_id = 1:(chosen_features_num)
            population(i, feat_perm(f_id)) = 1;
        end
    end

    options = gaoptimset(@ga);
    options.PopulationType = 'doubleVector';
    options.InitialPopulation = population;
    options.Generations = 100;
    options.useParallel = 'true';

    intcon = (1:total_features * 3);
    nonlinearcon = @(x) nonlcon(x);

    % run the genetic algoritm

    feats = ga(...
        @(x) fitnessall(...
            x, targets, features_ds, ...
            strcat(int2str(act),'vsall'), time_interval ...
        ), total_features * 3, [], [], [], [], ...
        zeros(1,total_features * 3), ones(1,total_features * 3), ...
        nonlinearcon, intcon, options);

    onevsall_all{time_interval}{act}.features = genes2feat(feats);
    
    for i = 1:length(onevsall_all{time_interval}{act}.nets)
        if isequal(onevsall_all{time_interval}{act}.nets{i}.features, ...
                onevsall_all{time_interval}{act}.features)
            onevsall_all{time_interval}{act}.best_net = ...
                onevsall_all{time_interval}{act}.nets{i};
            break;
        end
    end
end
