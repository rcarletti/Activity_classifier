global fcc_all;
global onevsall_all;

fcc_all = struct;
fcc_all.net = struct;

retrievebestfeatures(features_ds);

function [] = retrievebestfeatures(features_ds)
    
    global total_features;
    global chosen_features_num;
    global fcc_all;
    
    %build targets
    O = ones(1,30);
    Z = zeros(1,30);
    targets = [O Z Z Z ; Z O Z Z ; Z Z O Z ; Z Z Z O ];
    
    %set up things for the genetic algorithm 
    %this time we consider 33 features, 11 features for each sensor
    population_size = 100;
    population_all = zeros(100, total_features * 3);
    rng('shuffle');


    %generate random population (pop_size x total_features array, features set to 1 are
    %the chosen features for that individual
    for i=1:population_size
        feat_perm = randperm(total_features * 3, chosen_features_num);
        for f_id =1:(chosen_features_num)
            population_all(i,feat_perm(f_id)) = 1;
        end
    end

    options = gaoptimset(@ga);
    options.PopulationType = 'doubleVector';
    options.InitialPopulation = population_all;
    options.useParallel = 'true';

    intcon = (1:total_features * 3);
    nonlinearcon = @(x)nonlcon(x);
    
    feats = ga(@(x) fitnessall(x,targets, features_ds, '4cc'), total_features * 3, [], [], [], [], ...
            zeros(1,total_features * 3), ones(1,total_features * 3), nonlinearcon, intcon, options);
        
    fcc_all.net.features = genes2feat(feats);
    
end