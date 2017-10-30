function plot_features(features_ds, feats, names, sensor, time_interval)
%PLOT_FEATURES Plot the selected features for the specified time interval

    figure;
    
    ff = zeros(length(feats), 40 * time_interval);
    for f_id = 1:length(feats)
        for a_id = 1:4
            for v_id = 1:(10 * time_interval)
                ff(f_id, (a_id - 1) * 10 * time_interval + v_id) = ...
                    dsgetfeature(features_ds, feats(f_id), sensor, a_id, v_id, time_interval);
            end
        end
        
        subplot(2,2,f_id);
        plot(ff(f_id, :));
        title(names(feats(f_id)));
    end
end
