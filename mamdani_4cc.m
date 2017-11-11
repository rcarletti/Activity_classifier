%% Perform Mamdani FIS classification for the four-class classifier

if ~exist('mamdani', 'var')
    mamdani = struct;
end

mamdani.fcc = eval_mamdani_fcc(fcc_ind, features_ds);

function mamdani = eval_mamdani_fcc(fcc, features_ds)
    % The chosen feature set corresponds to the independent sensors method
    % with 164s time interval.

    filename = 'Mamdani_4cc_ind_164s.fis';
    time_interval = 1;
    sensor_id = fcc{time_interval}.best_sensor.index;
    features = fcc{time_interval}.best_sensor.features;

    % Load Mamdani FIS

    mamdani.fis = readfis(filename);

    % Compute inputs and target outputs by using the feature set previously computed

    mamdani.inputs = zeros(4, 40 * time_interval);
    mamdani.targets = zeros(1, 40 * time_interval);
    
    for a_id = 1:4
        for v_id = 1:(10 * time_interval)
            for f_id = 1:4
                mamdani.inputs(f_id, (a_id - 1) * 10 * time_interval + v_id) = ...
                    dsgetfeature(features_ds, features(f_id), ...
                        sensor_id, a_id, v_id, time_interval);
            end

            mamdani.targets(1, (a_id - 1) * 10 * time_interval + v_id) = a_id;
        end
    end

    % Compute fuzzy output values

    mamdani.outputs = evalfis(mamdani.inputs, mamdani.fis)';
    mamdani.rms = rms(mamdani.outputs - mamdani.targets);
    
    % Plot target and output values

    plot(1:(40 * time_interval), mamdani.outputs, 1:(40 * time_interval), mamdani.targets);
    
end
