%% choose parameters for ANFIS - four class classifier (independent sensors)

for time_interval = [1,2,4]
    [fcc_ind{time_interval}.sugeno, fcc_ind{time_interval}.fis_input] = ...
        perform_sugeno(fcc_ind{time_interval}.best_sensor, features_ds, time_interval);
end

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
    nchk = (40 * time_interval) - ntrn;

    sugeno = struct;
    sugeno.training_data = input_perm(1:ntrn, 1:5);
    sugeno.validation_data = input_perm(ntrn+1:(40 * time_interval), 1:5);

    % generate and train the sugeno FIS

    nmfs = 2;
    epochs = 150;

    % generate initial FIS
    sugeno.genopt = genfisOptions('GridPartition');
    sugeno.genopt.NumMembershipFunctions = nmfs;
    sugeno.genopt.InputMembershipFunctionType = 'gbellmf';

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

    % plot output data
%    figure;
% 
%     subplot(2,2,1);
%     plot(1:ntrn, sugeno.fcc.training_data(:,5), '*r', 1:ntrn, sugeno.fcc.training_out(:,1), '*b');
%     legend('Training Data', 'ANFIS Output');
% 
%     subplot(2,2,2);
%     plot(1:nchk, sugeno.fcc.validation_data(:,5), '*r', 1:nchk, sugeno.fcc.validation_out(:,1), '*b');
%     legend('Training Data', 'ANFIS Output');
% 
%     subplot(2,2,[3,4]);
%     plot(1:epochs, sugeno.fcc.train_err, '.r', 1:epochs, sugeno.fcc.check_err, '*b');
%     legend('Training error', 'Validation error');
end
