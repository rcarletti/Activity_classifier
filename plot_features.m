%% four-class classifier

for tid = [1,2,4]
    plot_feat(features_ds, fcc_ind{tid}.best_sensor.features, features_names, ...
        fcc_ind{tid}.best_sensor.index, tid);
end

for tid = [1,2,4]
    plot_feat(features_ds, fcc_all{tid}.best_net.features, features_names, ...
        0, tid);
end

%% one-vs-all classifier

aid = 1; % activity to plot

for tid = [1,2,4]
    plot_feat(features_ds, onevsall_ind{tid}{aid}.best_sensor.features, features_names, ...
        onevsall_ind{tid}{aid}.best_sensor.index, tid);
end

for tid = [1,2,4]
    plot_feat(features_ds, onevsall_all{tid}{aid}.best_net.features, features_names, ...
        0, tid);
end

%% 
function plot_feat(features_ds, feats, names, sensor, time_interval)
%PLOT_FEATURES Plot the selected features for the specified time interval
    global total_features;

    figure;
    
    ff = zeros(length(feats), 40 * time_interval);
    for f_id = 1:length(feats)
        if sensor == 0
            s_id = ceil(feats(f_id) / total_features);
            feats(f_id) = mod(feats(f_id) - 1, total_features) + 1;
        else
            s_id = sensor;
        end
        
        for a_id = 1:4
            for v_id = 1:(10 * time_interval)
                ff(f_id, (a_id - 1) * 10 * time_interval + v_id) = ...
                    dsgetfeature(features_ds, feats(f_id), s_id, a_id, v_id, time_interval);
            end
        end
        
        subplot(2,2,f_id);
        plot(ff(f_id, :));
        title(names(feats(f_id)));
    end
end
