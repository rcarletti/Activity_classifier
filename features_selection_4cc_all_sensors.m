%% Perform best feature selection on a 4-class classifier using all sensors

global fcc_all;

fcc_all = cell(1,4);
for time_interval = [1,2,4]
    fcc_all{time_interval} = struct;
    fcc_all{time_interval}.nets = cell(1,0);
    retrievebestfeatures(features_ds,time_interval);
end

function [] = retrievebestfeatures(features_ds, time_interval)
    global chosen_features_num;
    global total_features;
    global fcc_all;
    
    % create targets

    Z = zeros(1, (10 * time_interval));
    O = ones(1, (10 * time_interval));

    targets = [O Z Z Z ; Z O Z Z ; Z Z O Z ; Z Z Z O ];
    
    % set up things for the genetic algorithm 
    % this time we consider 33 features, 11 features for each sensor

    population_size = 100;
    population_all = zeros(population_size, total_features * 3);
    rng('shuffle');

    % generate random population (pop_size x total_features) array
    % features set to 1 are the chosen features for that individual

    for i=1:population_size
        feat_perm = randperm(total_features * 3, chosen_features_num);
        for f_id =1:(chosen_features_num)
            population_all(i,feat_perm(f_id)) = 1;
        end
    end

    options = gaoptimset(@ga);
    options.PopulationType = 'doubleVector';
    options.InitialPopulation = population_all;
    options.Generations = 100;
    options.useParallel = 'true';

    intcon = (1:total_features * 3);
    nonlinearcon = @(x)nonlcon(x);
    
    feats = ga(@(x) fitnessall(x, targets, features_ds, '4cc', time_interval), ...
        total_features * 3, [], [], [], [], zeros(1,total_features * 3), ...
        ones(1,total_features * 3), nonlinearcon, intcon, options);
        
    fcc_all{time_interval}.features = genes2feat(feats);
    
    for i = 1:length(fcc_all{time_interval}.nets)
        if isequal(fcc_all{time_interval}.nets{i}.features, ...
                fcc_all{time_interval}.features)
            fcc_all{time_interval}.best_net = fcc_all{time_interval}.nets{i};
            break;
        end
    end
end
