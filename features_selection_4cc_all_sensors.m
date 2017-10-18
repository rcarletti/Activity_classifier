global total_features

%% set up things for neural networks

targets_4cc_all = [ones(1,30),  zeros(1,30), zeros(1,30), zeros(1,30);...
                   zeros(1,30), ones(1,30),  zeros(1,30), zeros(1,30);...
                   zeros(1,30), zeros(1,30), ones(1,30),  zeros(1,30);...
                   zeros(1,30), zeros(1,30), zeros(1,30), ones(1,30)];

%% genetic algorithm
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

intcon = (1:33);
nonlinearcon = @(x)nonlcon(x);

global best_all_4cc;
best_all_4cc = struct;

best_all_4cc.features = ga(@(x) fitnessall(x,targets_4cc_all, features_ds, '4cc'), total_features * 3, [], [], [], [], ...
            zeros(1,33), ones(1,33), nonlinearcon, intcon, options);

%%
        
j = 1;
app = zeros(1,4);
for i=1:total_features * 3
    if best_all_4cc.features(i) == 1
        app(j) = i;
        j = j+1;
    end
end

best_all_4cc.features = app;

