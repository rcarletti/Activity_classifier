load('data.mat');

global chosen_features_num;
global total_features;

chosen_features_num = 4;

%% filter and normalize data

filtered_s = filter_data(data);
%filtered_s = remove_mean(filtered_s);
%filtered_s = normalize_data(filtered_s);

%% extract features for each sensor 

[features_ds, features_names, total_features] = extract_features(filtered_s);

%% features selection for the 4 class classifier (independent sensors)

features_selection_4cc_independent_sensors
disp('-------------features selection four class classifier with independent sensors done-------------')

%% features selection for the 4 class classifier (all sensors)

features_selection_4cc_all_sensors
disp('-------------features selection four class classifier with all sensors done-------------')

%% features selection for the one-against-all classifier (independent sensors)

features_selection_onevsall_independent_sensors
disp('-------------features selection onevsall classifier with independent sensors done-------------')

%% features selection for the one-against-all classifier (all sensors)

features_selection_onevsall_all_sensors
disp('-------------features selection onevsall classifier with all sensors done-------------')

%% sugeno-type FIS using ANFIS - four class classifier - independents sensors
% use the best sensor computed previously
sugeno_4cc_independent_sensors

%% sugeno-type FIS using ANFIS - four class classifier - all sensors
sugeno_4cc_all_sensors

%% sugeno-type FIS using ANFIS - one against all classifier - independent sensors
% use the best sensor computed previously
sugeno_onevsall_independent_sensors

%% sugeno-type FIS using ANFIS - one against all classifier - all sensors
% use the best sensor computed previously
sugeno_onevsall_all_sensors

save('activity_class_workspace.mat');

%% mamdani-type FIS - four class classifier - independent sensors
% use the best sensor computed previously

mamdani_4cc_independent_sensors

%% Helper functions

function [filtered_s] = filter_data(data)
% Apply Savitzky-Golay filter to input data
    filtered_s = dsnew();

    for s_id = 1:3           %for each sensor
        for a_id = 1:4       %for each actovity
            for v_id  = 1:10 %for each volunteer
                fs = sgolayfilt(dsget(data, s_id, a_id, v_id), 4, 21);
                filtered_s = dsput(filtered_s, fs, s_id, a_id, v_id);
            end
        end
    end
end

function [filtered_s] = remove_mean(filtered_s)
% Remove mean from filtered dataset
    for s_id = 1:3
        for a_id = 1:4
            for v_id = 1:10
                ds = detrend(dsget(filtered_s, s_id, a_id, v_id), 'constant');
                filtered_s = dsput(filtered_s, ds, s_id, a_id, v_id);
            end
        end
    end
end

function [filtered_s] = normalize_data(filtered_s)
% Normalize filtered dataset
    for s_id = 1:3
        for a_id = 1:4
            for v_id = 1:10
                s = dsget(filtered_s, s_id, a_id, v_id);
                
                %compute standard deviation for each signal
                standard_dev = std(s);
                
                %divide the signal for the standard deviation
                normalized_signal = s / standard_dev;
                
                filtered_s = dsput(filtered_s, normalized_signal, s_id, a_id, v_id);     
            end
        end
    end
end

function [features_ds, features_names, total_features] = extract_features(filtered_ds)
% Extract features from every signal
    features_ds = dsnew();
    
    for s_id = 1:3
        for a_id = 1:4
            for v_id = 1:10
                feat_t = getfeatures(dsget(filtered_ds, s_id, a_id, v_id), 't');
                feat_f = getfeatures(dsget(filtered_ds, s_id, a_id, v_id), 'f');
                feat = [feat_t, feat_f];
                total_features = length(feat);
                features_ds = dsputfeatures(features_ds, feat, s_id, a_id, v_id);
            end
        end
    end

    features_names = ["min", "max", "mean", "std_dev","peak2rms", "peak2peak", ...
        "rssq", "occupied_band", "power", "meanfreq", "bandpower"];
end
