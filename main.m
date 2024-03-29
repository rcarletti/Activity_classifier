load('data.mat');

global chosen_features_num;
global total_features;

chosen_features_num = 4;

%% split data into time intervals

%create one dataset for each time interval [164s,82s,41s]
data_raw = cell(1,4);
for time_intervals = [1,2,4]
    data_raw{time_intervals} = dsnew(time_intervals);
    data_raw = fill_data(data, data_raw, time_intervals);
end

%% filter data

Fpass = 0.4;
Fstop = 0.41;
Apass = 1;
Astop = 40;

df = designfilt('lowpassfir', 'DesignMethod', 'kaiserwin', ...
  'PassbandFrequency', Fpass, 'StopbandFrequency', Fstop, ...
  'PassbandRipple', Apass, 'StopbandAttenuation', Astop);

filtered_s = cell(1,4);
for time_intervals = [1,2,4]
        filtered_s{time_intervals} = dsnew(time_intervals);
        %plot(dsget(data_raw,2,1,1,1));
        %hold on
        filtered_s = filter_data(filtered_s, df, data_raw, time_intervals);
        %plot(dsget(filtered_s,2,1,1,1));
        filtered_s = remove_mean(filtered_s, time_intervals);
        %plot(dsget(filtered_s,2,1,1,1));
        filtered_s = normalize_data(filtered_s, time_intervals);
        %plot(dsget(filtered_s,2,1,1,1));
end

%% extract features for each sensor and time interval

features_ds = cell(1,4);
for time_intervals = [1,2,4]
        features_ds{time_intervals} = dsnew(time_intervals);
        [features_ds, features_names, total_features] = ...
            extract_features(filtered_s,features_ds, time_intervals);
end

%% features selection for the 4 class classifier (independent sensors)

features_selection_4cc_independent_sensors
disp('-------------features selection four class classifier with independent sensors done-------------')

disp(datetime('now'));
save('refactored_ws');

%% features selection for the 4 class classifier (all sensors)

features_selection_4cc_all_sensors
disp('-------------features selection four class classifier with all sensors done-------------')

disp(datetime('now'));
save('refactored_ws');

%% features selection for the one-against-all classifier (independent sensors)

features_selection_onevsall_independent_sensors
disp('-------------features selection onevsall classifier with independent sensors done-------------')

disp(datetime('now'));
save('refactored_ws');

%% features selection for the one-against-all classifier (all sensors)

features_selection_onevsall_all_sensors
disp('-------------features selection onevsall classifier with all sensors done-------------')

disp(datetime('now'));
save('refactored_ws');

%% sugeno-type FIS using ANFIS - four class classifier - independents sensors
% use the best sensor computed previously

sugeno_4cc_independent_sensors

disp(datetime('now'));

%% sugeno-type FIS using ANFIS - four class classifier - all sensors

sugeno_4cc_all_sensors

disp(datetime('now'));

%% sugeno-type FIS using ANFIS - one against all classifier - independent sensors
% use the best sensor computed previously

sugeno_onevsall_independent_sensors

disp(datetime('now'));

%% sugeno-type FIS using ANFIS - one against all classifier - all sensors
% use the best sensor computed previously

sugeno_onevsall_all_sensors

disp(datetime('now'));

%% mamdani-type FIS - four class classifier - independent sensors
% use the best sensor computed previously

%mamdani_4cc_independent_sensors

%% Helper functions

function [filtered_s] = filter_data(filtered_s, df, data_raw, time_interval)
% Apply Savitzky-Golay filter to input data

    for s_id = 1:3           %for each sensor
        for a_id = 1:4       %for each actovity
            for v_id  = 1:(10 * time_interval) %for each volunteer
                y1 = filter(df, dsget(data_raw, s_id, a_id, v_id, time_interval));
                filtered_s = dsput(filtered_s, y1, s_id, a_id, v_id, time_interval);
            end
        end
    end
end

function [filtered_s] = remove_mean(filtered_s, time_interval)
% Remove mean from filtered dataset
    for s_id = 1:3
        for a_id = 1:4
            for v_id = 1:(10 * time_interval)
                ds = detrend(dsget(filtered_s, s_id, a_id, v_id, time_interval), 'constant');
                filtered_s = dsput(filtered_s, ds, s_id, a_id, v_id,time_interval);
            end
        end
    end
end

function [filtered_s] = normalize_data(filtered_s, time_interval)
% Normalize filtered dataset
    for s_id = 1:3
        for a_id = 1:4
            for v_id = 1:(10 * time_interval)
                s = dsget(filtered_s, s_id, a_id, v_id, time_interval);
                
                %compute standard deviation for each signal
                standard_dev = std(s);
                
                %divide the signal for the standard deviation
                normalized_signal = s / standard_dev;
                
                filtered_s = dsput(filtered_s, normalized_signal, s_id, a_id, v_id, time_interval);     
            end
        end
    end
end

function [features_ds, features_names, total_features] = extract_features(filtered_ds,features_ds, time_interval)
% Extract features from every signal
    
    for s_id = 1:3
        for a_id = 1:4
            for v_id = 1:(10 * time_interval)
                feat_t = getfeatures(dsget(filtered_ds, s_id, a_id, v_id, time_interval), 't');
                feat_f = getfeatures(dsget(filtered_ds, s_id, a_id, v_id, time_interval), 'f');
                feat = [feat_t, feat_f];
                total_features = length(feat);
                features_ds = dsputfeatures(features_ds, feat, s_id, a_id, v_id, time_interval);
            end
        end
    end

    features_names = [ "min", "max", "median", "rms", "meanabs", "sumabs", ...
        "sumabsdiff", "peak2rms", "peak2peak", "rssq", "zc", "mue", "mle", ...
        "obw", "pobw", "meanfreq", "medfreq", "bandpower", "sumfft", ...
        "powDC", "npeaks", "avgpeakdist", "sumpsd" ];
end

function [data_raw] = fill_data(data, data_raw, time_interval)
% Split data in the given time intervals

    for s_id = 1:3
        for a_id = 1:4
            for v_id = 1:10
                s = dsgetdata(data, s_id, a_id, v_id);

                for t = 1:time_interval
                    a = 1+(length(s)/time_interval) * (t-1);
                    b = (length(s)/time_interval) * t;
                    data_raw = dsput(data_raw, s(a:b), s_id, a_id, ...
                        (v_id - 1) * time_interval + t, time_interval);
                end

            end
        end
    end
end
