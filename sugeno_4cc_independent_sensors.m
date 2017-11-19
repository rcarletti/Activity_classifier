%% choose parameters for ANFIS - four class classifier (independent sensors)

if ~exist('sugeno', 'var')
    sugeno = struct;
end

if ~isfield(sugeno, 'fcc_ind')
    sugeno.fcc_ind = cell(1,4);
end

time_interval = 4;

for i = 1:100
    [fis, inputs] = perform_sugeno(fcc_ind{time_interval}.best_sensor, ...
        features_ds, time_interval);
    
    if ~isfield(sugeno.fcc_ind{time_interval}, 'fis') || ...
       fis.error < sugeno.fcc_ind{time_interval}.fis.error
        sugeno.fcc_ind{time_interval}.fis = fis;
        sugeno.fcc_ind{time_interval}.inputs = inputs;
    end
end

plot_sugeno(sugeno.fcc_ind{time_interval}.fis);

function [sugeno, input] = perform_sugeno(sensor, features_ds, time_interval)

    % create the matrix with features for each couple activity-volunteer
    % the data are arranged in [ feat1, feat2, feat3, feat4, activity, volunteer ]

    input = zeros(40 * time_interval, 6);
    for a_id = 1:4
        for v_id = 1:(10 * time_interval)
            for f_id = 1:4
                input((a_id - 1) * 10 * time_interval + v_id, f_id) = dsgetfeature(features_ds, ...
                    sensor.features(f_id), sensor.index, a_id, v_id, time_interval);
            end
            input((a_id - 1) * 10 * time_interval + v_id, 5) = a_id;
            input((a_id - 1) * 10 * time_interval + v_id, 6) = v_id;
        end
    end

    % shuffle columns
    
    input_perm = input(randperm(size(input,1)),:);

    % select ANFIS data - 70%-30% split

    ntrn = floor(40 * 0.7 * time_interval);

    sugeno = struct;
    sugeno.training_data = input_perm(1:ntrn, 1:5);
    sugeno.validation_data = input_perm(ntrn+1:(40 * time_interval), 1:5);

    % generate and train the sugeno FIS
    epochs = 50;

    % generate initial FIS
    sugeno.genopt = genfisOptions('SubtractiveClustering', ...
                                  'ClusterInfluenceRange', 0.3);

    % set FIS options
    sugeno.fisopt = anfisOptions('EpochNumber', epochs, 'OptimizationMethod', 1, 'InitialFIS', ...
        genfis(sugeno.training_data(:,1:4), sugeno.training_data(:,5), sugeno.genopt));
    sugeno.fisopt.DisplayErrorValues = 0;
    sugeno.fisopt.DisplayStepSize = 0;
    sugeno.fisopt.ValidationData = sugeno.validation_data;

    % run ANFIS
    [sugeno.fis, sugeno.train_err, ~, sugeno.check_fis, sugeno.check_err] = ...
        anfis(sugeno.training_data, sugeno.fisopt);

    % compute fuzzy output values
    sugeno.training_out = evalfis(sugeno.training_data(:,1:4), sugeno.fis);
    sugeno.validation_out = evalfis(sugeno.validation_data(:,1:4), sugeno.fis);
    
    % compare crisp validation output values
    sugeno.crisp_out = round(sugeno.validation_out);
    sugeno.crisp_out(sugeno.crisp_out > 4) = 4;
    sugeno.crisp_out(sugeno.crisp_out < 1) = 1;
    
    % compute accuracy and error rate
    sugeno.error = sum(abs(sugeno.crisp_out - sugeno.validation_data(:,5)) > 0) ...
        / length(sugeno.crisp_out);
    sugeno.accuracy = 1 - sugeno.error;
end
