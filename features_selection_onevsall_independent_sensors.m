global neural_networks_1vsall;
global neural_networks_2vsall;
global neural_networks_3vsall;
global neural_networks_4vsall;


%create targets 
%class 1 vs all
targets_onevsall{1} = [ones(1,10), zeros(1,10), zeros(1,10), zeros(1,10);
                        zeros(1,10), ones(1,10), ones(1,10), ones(1,10)];
               
%class 2 vs all
targets_onevsall{2} = [zeros(1,10), ones(1,10), zeros(1,10), zeros(1,10);
                        ones(1,10), zeros(1,10), ones(1,10), ones(1,10)];

%class 3 vs all
targets_onevsall{3} = [zeros(1,10), zeros(1,10), ones(1,10), zeros(1,10);
                        ones(1,10), ones(1,10), zeros(1,10), ones(1,10)];
                  
%class 4 vs all
targets_onevsall{4} = [zeros(1,10), zeros(1,10), zeros(1,10), ones(1,10);
                        ones(1,10), ones(1,10), ones(1,10), zeros(1,10)];
                       

%% create (n,k) neural networks for each sensor for each activity

neural_networks_1vsall = cell(1,3);
for s_id =1:3
    neural_networks_1vsall{s_id} = createandtrainnn(s_id, inputs, ...
                                targets_onevsall{1}, nets_num);
end

neural_networks_2vsall = cell(1,3);
for s_id =1:3
    neural_networks_2vsall{s_id} = createandtrainnn(s_id, inputs, ...
                                targets_onevsall{2}, nets_num);
end

neural_networks_3vsall = cell(1,3);
for s_id =1:3
    neural_networks_3vsall{s_id} = createandtrainnn(s_id, inputs, ...
                                targets_onevsall{3}, nets_num);
end

neural_networks_4vsall = cell(1,3);
for s_id =1:3
    neural_networks_4vsall{s_id} = createandtrainnn(s_id, inputs, ...
                                targets_onevsall{4}, nets_num);
end

%% genetic algorithm

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

best_features_1vsall = cell(1,3);
for s_id=1:3
    %get the best set of features for each sensor
    best_features_1vsall{1,s_id}{1} = ga(@(x) fitnessfunction(x, s_id, '1vsall'), total_features, [], [], [], [], ...
            zeros(1,11), ones(1,11), nonlinearcon, intcon, options);
end

best_features_2vsall = cell(1,3);
for s_id=1:3
    %get the best set of features for each sensor
    best_features_2vsall{1,s_id}{1} = ga(@(x) fitnessfunction(x, s_id, '2vsall'), total_features, [], [], [], [], ...
            zeros(1,11), ones(1,11), nonlinearcon, intcon, options);
end

best_features_3vsall = cell(1,3);
for s_id=1:3
    %get the best set of features for each sensor
    best_features_3vsall{1,s_id}{1} = ga(@(x) fitnessfunction(x, s_id, '3vsall'), total_features, [], [], [], [], ...
            zeros(1,11), ones(1,11), nonlinearcon, intcon, options);
end

best_features_4vsall = cell(1,3);
for s_id=1:3
    %get the best set of features for each sensor
    best_features_4vsall{1,s_id}{1} = ga(@(x) fitnessfunction(x, s_id, '4vsall'), total_features, [], [], [], [], ...
            zeros(1,11), ones(1,11), nonlinearcon, intcon, options);
end

%% compute accuracy for each sensor for each classifier

%1 vs all
for s_id=1:3
    feat = best_features_1vsall{1,s_id}{1};
    net = getnetworkbyfeatures(feat, s_id,'1vsall');
    best_features_1vsall{1,s_id}{2} = 1-net.conf;
end

%2 vs all
for s_id=1:3
    feat = best_features_2vsall{1,s_id}{1};
    net = getnetworkbyfeatures(feat, s_id,'2vsall');
    best_features_2vsall{1,s_id}{2} = 1-net.conf;
end

%3 vs all
for s_id=1:3
    feat = best_features_3vsall{1,s_id}{1};
    net = getnetworkbyfeatures(feat, s_id,'3vsall');
    best_features_3vsall{1,s_id}{2} = 1-net.conf;
end

%4 vs all
for s_id=1:3
    feat = best_features_4vsall{1,s_id}{1};
    net = getnetworkbyfeatures(feat, s_id,'4vsall');
    best_features_4vsall{1,s_id}{2} = 1-net.conf;
end

%% choose the best sensor for each activity 

%1 vs all
max = 0;
s_index = 1;
for s_id=1:3
    if best_features_1vsall{1,s_id}{2} > max
        max = best_features_1vsall{1,s_id}{2};
        s_index = s_id;
    end
end

best_sensor_1vsall.index = s_index;
best_sensor_1vsall.features = best_features_1vsall{1,s_index}{1};
best_sensor_1vsall.accuracy = max;

%2 vs all
max = 0;
s_index = 1;
for s_id=1:3
    if best_features_2vsall{1,s_id}{2} > max
        max = best_features_2vsall{1,s_id}{2};
        s_index = s_id;
    end
end

best_sensor_2vsall.index = s_index;
best_sensor_2vsall.features = best_features_1vsall{1,s_index}{1};
best_sensor_2vsall.accuracy = max;

%3 vs all
max = 0;
s_index = 1;
for s_id=1:3
    if best_features_3vsall{1,s_id}{2} > max
        max = best_features_3vsall{1,s_id}{2};
        s_index = s_id;
    end
end

best_sensor_3vsall.index = s_index;
best_sensor_3vsall.features = best_features_1vsall{1,s_index}{1};
best_sensor_3vsall.accuracy = max;

%4 vs all
max = 0;
s_index = 1;
for s_id=1:3
    if best_features_4vsall{1,s_id}{2} > max
        max = best_features_4vsall{1,s_id}{2};
        s_index = s_id;
    end
end

best_sensor_4vsall.index = s_index;
best_sensor_4vsall.features = best_features_1vsall{1,s_index}{1};
best_sensor_4vsall.accuracy = max;



save('activity_class_workspace.mat');
