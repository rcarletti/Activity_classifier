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

%% features selection for the 4 class classifier (independent sensors)

features_selection_4cc_independent_sensors

%% features selection for the one-against-all classifier (independent sensors)

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
                       
%create inputs
inputs_1vsall = cell(1,3);
inputs_1vsall{1} = zeros(4, 40, size(C,1));        %sensor #1
inputs_1vsall{2} = zeros(4, 40, size(C,1));        %sensor #2
inputs_1vsall{3} = zeros(4, 40, size(C,1));        %sensor #3

inputs_2vsall = cell(1,3);
inputs_2vsall{1} = zeros(4, 40, size(C,1));        %sensor #1
inputs_2vsall{2} = zeros(4, 40, size(C,1));        %sensor #2
inputs_2vsall{3} = zeros(4, 40, size(C,1));        %sensor #3

inputs_3vsall = cell(1,3);
inputs_3vsall{1} = zeros(4, 40, size(C,1));        %sensor #1
inputs_3vsall{2} = zeros(4, 40, size(C,1));        %sensor #2
inputs_3vsall{3} = zeros(4, 40, size(C,1));        %sensor #3

inputs_4vsall = cell(1,3);
inputs_4vsall{1} = zeros(4, 40, size(C,1));        %sensor #1
inputs_4vsall{2} = zeros(4, 40, size(C,1));        %sensor #2
inputs_4vsall{3} = zeros(4, 40, size(C,1));        %sensor #3

%class 1 inputs

for s_id = 1:3
    for t_id = 1:size(C,1)
        for a_id = 1:4
            for v_id = 1:10
                for f_id = 1:chosen_features_num
                    inputs_1vsall{s_id}(f_id, ((a_id-1) * 10) + v_id, t_id) = ...
                        dsgetfeature(features_ds, C(t_id, f_id), s_id, a_id, v_id);
                end
            end
        end
    end
end

%class 2 targets

for s_id =1:3 
    for t_id = 1:size(C,1)
        for v_id = 1:10
            for f_id = 1:chosen_features_num
                inputs_2vsall{s_id}(f_id, v_id,t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 2, v_id);
                inputs_2vsall{s_id}(f_id,(10 + v_id),t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 1, v_id);
                inputs_2vsall{s_id}(f_id,(20 + v_id),t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 3, v_id);
                inputs_2vsall{s_id}(f_id,(30 + v_id),t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 4, v_id);
            end
            
        end
    end
end

%class 3 inputs

for s_id =1:3 
    for t_id = 1:size(C,1)
        for v_id = 1:10
            for f_id = 1:chosen_features_num
                inputs_3vsall{s_id}(f_id, v_id,t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 3, v_id);
                inputs_3vsall{s_id}(f_id,(10 + v_id),t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 1, v_id);
                inputs_3vsall{s_id}(f_id,(20 + v_id),t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 2, v_id);
                inputs_3vsall{s_id}(f_id,(30 + v_id),t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 4, v_id);
            end
            
        end
    end
end

%%class 4 inputs

for s_id =1:3 
    for t_id = 1:size(C,1)
        for v_id = 1:10
            for f_id = 1:chosen_features_num
                inputs_4vsall{s_id}(f_id, v_id,t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 4, v_id);
                inputs_4vsall{s_id}(f_id,(10 + v_id),t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 1, v_id);
                inputs_4vsall{s_id}(f_id,(20 + v_id),t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 2, v_id);
                inputs_4vsall{s_id}(f_id,(30 + v_id),t_id) = ...
                    dsgetfeature(features_ds, C(t_id, f_id), s_id, 3, v_id);
            end
        end
    end
end


%% create (n,k) neural networks for each sensor for each activity

neural_networks_1vsall = cell(1,3);
for s_id =1:3
    neural_networks_1vsall{s_id} = createandtrainnn(s_id, inputs_1vsall, ...
                                targets_onevsall{1}, nets_num);
end

neural_networks_2vsall = cell(1,3);
for s_id =1:3
    neural_networks_2vsall{s_id} = createandtrainnn(s_id, inputs_2vsall, ...
                                targets_onevsall{2}, nets_num);
end

neural_networks_3vsall = cell(1,3);
for s_id =1:3
    neural_networks_3vsall{s_id} = createandtrainnn(s_id, inputs_3vsall, ...
                                targets_onevsall{3}, nets_num);
end

neural_networks_4vsall = cell(1,3);
for s_id =1:3
    neural_networks_4vsall{s_id} = createandtrainnn(s_id, inputs_4vsall, ...
                                targets_onevsall{4}, nets_num);
end

