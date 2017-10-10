load('data.mat');

filtered_s = dsnew();
normalized_s = dsnew();
chosen_features_num = 4;

global total_features;

%% filter data

for s_id = 1:3                          %for each sensor
    for a_id = 1:4                      %for each actovity
        for v_id  = 1:10                %for each volunteer
            fs = sgolayfilt(dsget(data,s_id,a_id,v_id),4,21);
            filtered_s = dsput(filtered_s,fs,s_id,a_id,v_id);
        end
    end
end

%plot(dsget(data,1,1,1));
%hold on 
%plot(dsget(filtered_s,1,1,1));

%% subtract mean value

for s_id = 1:3
    for a_id = 1:4
        for v_id = 1:10
            ds = detrend(dsget(filtered_s, s_id, a_id, v_id),'constant');
            filtered_s = dsput(filtered_s,ds, s_id,a_id,v_id);
        end
    end
end

%plot(dsget(filtered_s,1,1,1));


%% z-normalization

for s_id = 1:3
    for a_id = 1:4
        for v_id = 1:10
            s = dsget(filtered_s,s_id,a_id,v_id);
            %compute standard deviation for each signal
            standard_dev = std(s);
            %divide the signal for the standard deviation
            normalized_signal = s/standard_dev;
            normalized_s = dsput(normalized_s, normalized_signal, s_id,...
                a_id, v_id);     
        end
    end
end

%plot(dsget(normalized_s,1,1,1));

%% extract features for each sensor
features_ds = dsnew();

for s_id = 1:3
    for a_id = 1:4
        for v_id = 1:10
            feat_t = getfeatures(dsget(normalized_s,s_id,a_id,v_id), 't');
            feat_f = getfeatures(dsget(normalized_s,s_id,a_id,v_id), 'f');
            feat = [feat_t, feat_f];
            total_features = length(feat);
            features_ds = dsputfeatures(features_ds,feat,s_id,a_id,v_id);
        end
    end
end

%% create and train neural nwtworks (4 features)

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


%% create (n,k) neural networks for each sensor and train them
global neural_networks;

neural_networks = cell(1,3);
neural_networks{1} = createandtrainnn(1, inputs, targets, nets_num);
neural_networks{2} = createandtrainnn(2, inputs, targets, nets_num);
neural_networks{3} = createandtrainnn(3, inputs, targets, nets_num);

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

x_s1 = ga(@(x) fitnessfunction(x, 1), total_features, [], [], [], [], ...
    zeros(1,11), ones(1,11), nonlinearcon, intcon, options)

