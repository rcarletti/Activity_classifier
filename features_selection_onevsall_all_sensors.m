%% Perform best feature selection on a 1-vs-all classifier with all sensors
global onevsall_all;

onevsall_all = cell(1,4);
for time_interval = [1,2,4]
    onevsall_all{time_interval} = cell(1,4);

    for i = 1:4
        onevsall_all{time_interval}{i} = struct;
        onevsall_all{time_interval}{i}.net = struct;
        retrievebestfeatures(act,features_ds, time_interval);
    end
end

function [] = retrievebestfeatures(act, features_ds, time_interval)
    
    global total_features;
    global chosen_features_num;
    global onevsall_all;
    
    % generate targets for class act
   
    targets = zeros(2, 120 * time_interval);
    targets(2,:) = ones(1, 120 * time_interval);
    targets(1, (1:30 * time_interval) + (act-1)* 10 * time_interval) = ones(1,30 * time_interval);
    targets(2, (1:30 * time_interval) + (act-1)* 10 * time_interval) = zeros(1,30 * time_interval);

    % set up the GA, this time we consider 33 features, 11 features for each sensor

    population_size = 100;
    population_all = zeros(100, total_features * 3);

    % generate random population (pop_size x total_features) array
    % features set to 1 are the chosen features for that individual

    for i = 1:population_size
        feat_perm = randperm(total_features * 3, chosen_features_num);
        for f_id = 1:(chosen_features_num)
            population_all(i,feat_perm(f_id)) = 1;
        end
    end

    options = gaoptimset(@ga);
    options.PopulationType = 'doubleVector';
    options.InitialPopulation = population_all;
    options.useParallel = 'true';

    intcon = (1:total_features * 3);
    nonlinearcon = @(x) nonlcon(x);

    % run the genetic algoritm

    feats = ga(...
        @(x) fitnessall(...
            x, targets, features_ds, ...
            strcat(int2str(i),'vsall'), time_interval ...
        ), total_features * 3, [], [], [], [], ...
        zeros(1,total_features * 3), ones(1,total_features * 3), ...
        nonlinearcon, intcon, options);

    onevsall_all{time_interval}{act}.net.features = genes2feat(feats);

end
