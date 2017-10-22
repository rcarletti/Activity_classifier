%% create targets
Z = zeros(1,30);
O = ones(1,30);

targets_onevsall{1} = [O Z Z Z; Z O O O];
targets_onevsall{2} = [Z O Z Z; O Z O O];
targets_onevsall{3} = [Z Z O Z; O O Z O];
targets_onevsall{4} = [Z Z Z O; O O O Z];
  
%% set-up the genetic algorithm

function [sensor, feats, nets] = retrievebestfeatures(class_num, targets, inputs, features_ds)
%this time we consider 33 features, 11 features for each sensor
population_size = 100;
population_all = zeros(100, total_features * 3);

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

global best_onevsall_all;
best_onevsall_all = cell(1,4);
for i=1:4
    best_onevsall_all{i} = struct;
end

%% run the genetic algoritm

for i=1:4
    best_onevsall_all{i}.features = ga(@(x) fitnessall(x,...
            targets_onevsall_all{i},...
            features_ds, ...
            strcat(int2str(i),'vsall')), ...
            total_features * 3,...
            [], [], [], [], ...
            zeros(1,33), ones(1,33), ...
            nonlinearcon, ...
            intcon, ...
            options);
end

%%

for i=1:4
    j = 1;
    app = zeros(1,4);
    for k=1:total_features * 3
        if best_onevsall_all{1,i}.features(k) == 1
            app(j) = k;
            j = j+1;
        end
    end
    
    best_onevsall_all{1,i}.features = app
        
end
