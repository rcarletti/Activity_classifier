%% create and train neural networks (4 class classifier)

targets = [ones(1,10),  zeros(1,10), zeros(1,10), zeros(1,10);...
           zeros(1,10), ones(1,10),  zeros(1,10), zeros(1,10);...
           zeros(1,10), zeros(1,10), ones(1,10),  zeros(1,10);...
           zeros(1,10), zeros(1,10), zeros(1,10), ones(1,10)];

global C;
global nets_num;
% find (n,k) combinations of features
features_positions = [1:total_features];
C = nchoosek(features_positions,chosen_features_num);
nets_num = nchoosek(total_features, chosen_features_num);

% create input vectors
inputs = cell(1,3);
inputs{1} = zeros(4, 40, size(C,1));        %sensor #1
inputs{2} = zeros(4, 40, size(C,1));        %sensor #2
inputs{3} = zeros(4, 40, size(C,1));        %sensor #3

for s_id = 1:3
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


%% create (n,k) neural networks for each sensor and train them (4 class classifier)
global neural_networks;

neural_networks = cell(1,3);
neural_networks{1} = createandtrainnn(1, inputs, targets, nets_num);
neural_networks{2} = createandtrainnn(2, inputs, targets, nets_num);
neural_networks{3} = createandtrainnn(3, inputs, targets, nets_num);

%% genetic algorithm (4 class classifier)

population_size = 100;
population = zeros(100, total_features);
rng('shuffle');

%generate random population (pop_size x total_features array, features set to 1 are
%the chosen features for that individual
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

intcon = (1:11);
nonlinearcon = @(x)nonlcon(x);

best_features_4cc = cell(1,3);

for s_id=1:3
    %get the best set of features for each sensor
    best_features_4cc{1,s_id}{1} = ga(@(x) fitnessfunction(x, s_id), total_features, [], [], [], [], ...
            zeros(1,11), ones(1,11), nonlinearcon, intcon, options);
end


%% compute accuracy for each sensor (4 class classifier)
for s_id=1:3
    feat = best_features_4cc{1,s_id}{1};
    net = getnetworkbyfeatures(feat, s_id);
    best_features_4cc{1,s_id}{2} = 1-net.conf;
end

%% choose the best sensor for the four class classifier


max = 0;
s_index = 1;
for s_id=1:3
    if best_features_4cc{1,s_id}{2} > max
        max = best_features_4cc{1,s_id}{2};
        s_index = s_id;
    end
end

best_sensor_4cc.index = s_index;
best_sensor_4cc.features = best_features_4cc{1,s_index}{1};
best_sensor_4cc.accuracy = max;


