%% Perform Mamdani FIS classification for the one-against-all classifier

if ~exist('mamdani', 'var')
    mamdani = struct;
end

mamdani.onevsall = cell(1,4);

%%
mamdani.onevsall{1} = eval_mamdani_onevsall('Mamdani_1vsAll_all_164s.fis', 1, ...
    onevsall_all{1}{1}.features, features_ds, 0, 1);

%%
mamdani.onevsall{2} = eval_mamdani_onevsall('Mamdani_2vsAll_all_41s.fis', 2, ...
    onevsall_all{4}{2}.features, features_ds, 0, 4);

%%
mamdani.onevsall{3} = eval_mamdani_onevsall('Mamdani_3vsAll_all_82s.fis', 3, ...
    onevsall_all{2}{3}.features, features_ds, 0, 2);

%%
mamdani.onevsall{4} = eval_mamdani_onevsall('Mamdani_4vsAll_ind_82s.fis', 4, ...
    onevsall_ind{2}{4}.best_sensor.features, features_ds, ...
    onevsall_ind{2}{4}.best_sensor.index, 2);

%%
function mamdani = eval_mamdani_onevsall(filename, act, features, features_ds, ...
                                         sensor_id, time_interval)
    global total_features
                                     
    % Load Mamdani FIS

    mamdani.fis = readfis(filename);

    % Compute inputs and target outputs by using the feature set previously computed

    mamdani.inputs = zeros(4, 40 * time_interval);
    mamdani.targets = zeros(1, 40 * time_interval);
    
    for a_id = 1:4
        for v_id = 1:(10 * time_interval)
            for f_id = 1:4
                if sensor_id == 0
                    feat = mod(features(f_id) - 1, total_features) + 1;
                    s_id = ceil(features(f_id) / total_features);
                else
                    feat = features(f_id);
                    s_id = sensor_id;
                end
                
                mamdani.inputs(f_id, (a_id - 1) * 10 * time_interval + v_id) = ...
                   dsgetfeature(features_ds, feat, s_id, a_id, v_id, time_interval);
            end

            mamdani.targets(1, (a_id - 1) * 10 * time_interval + v_id) = ...
                double(a_id == act);
        end
    end

    % Compute fuzzy output values

    mamdani.outputs = evalfis(mamdani.inputs, mamdani.fis)';
    mamdani.rms = rms(mamdani.outputs - mamdani.targets);
    mamdani.crisp = (mamdani.outputs > 0.5);
    
    % Plot target and output values
    figure;
    plot(1:(40 * time_interval), mamdani.crisp, '*b', ...
         1:(40 * time_interval), mamdani.targets, '+r', ...
         1:(40 * time_interval), mamdani.outputs);
    
end
