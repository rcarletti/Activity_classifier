%% ANFIS one-vs-all

seed = floor(rand * 10000);

sugeno_1vsall = anfis_onevsall(best_sensor_1vsall, 1, features_ds, seed);
sugeno_2vsall = anfis_onevsall(best_sensor_2vsall, 2, features_ds, seed);
sugeno_3vsall = anfis_onevsall(best_sensor_3vsall, 3, features_ds, seed);
sugeno_4vsall = anfis_onevsall(best_sensor_4vsall, 4, features_ds, seed);

%% Local functions

function [sugeno] = anfis_onevsall(best_sensor, act, features_ds, seed)
% choose parameters for ANFIS - one-vs-all classifier (independent sensors)

    %create the matrix with features for each couple activity-volunteer
    %the data are arranged in [feat1,feat2,feat3,feat4,activity,volunteer]

    a_v_matrix = zeros(40,6);
    for a_id = 1:4
        for v_id = 1:10
            for f_id = 1:4
                a_v_matrix((a_id -1) *10 + v_id, f_id) = dsgetfeature(features_ds,...
                    best_sensor.features(f_id), best_sensor.index, a_id, v_id);
            end
            a_v_matrix((a_id -1) * 10 + v_id, 5) = double(a_id == act);
            a_v_matrix((a_id -1) * 10 + v_id, 6) = v_id;
        end
    end

    %shuffle columns

    rng(seed);
    a_v_matrix_perm = a_v_matrix(randperm(size(a_v_matrix,1)),:);

    % select ANFIS data - 70%-30% split

    ntrn = floor(40*0.7);
    nchk = 40-ntrn;

    sugeno.onevsall.training_data = a_v_matrix_perm(1:ntrn, 1:5);
    sugeno.onevsall.validation_data = a_v_matrix_perm(ntrn+1:40, 1:5);

    % generate and train the sugeno FIS

    nmfs = 2;
    epochs = 150;

    % generate initial FIS
    sugeno.onevsall.genopt = genfisOptions('GridPartition');
    sugeno.onevsall.genopt.NumMembershipFunctions = nmfs;
    sugeno.onevsall.genopt.InputMembershipFunctionType = 'gbellmf';

    % set FIS options
    sugeno.onevsall.fisopt = anfisOptions('EpochNumber', epochs, 'OptimizationMethod', 1, 'InitialFIS', ...
        genfis(sugeno.onevsall.training_data(:,1:4), sugeno.onevsall.training_data(:,5), sugeno.onevsall.genopt));
    sugeno.onevsall.fisopt.DisplayErrorValues = 0;
    sugeno.onevsall.fisopt.DisplayStepSize = 0;
    sugeno.onevsall.fisopt.ValidationData = sugeno.onevsall.validation_data;

    % run ANFIS
    [sugeno.onevsall.fis, sugeno.onevsall.train_err, ~, sugeno.onevsall.check_fis, sugeno.onevsall.check_err] = ...
        anfis(sugeno.onevsall.training_data, sugeno.onevsall.fisopt);

    % compute fuzzy output values
    sugeno.onevsall.training_out = evalfis(sugeno.onevsall.training_data(:,1:4), sugeno.onevsall.fis);
    sugeno.onevsall.validation_out = evalfis(sugeno.onevsall.validation_data(:,1:4), sugeno.onevsall.fis);

    % plot output data
    figure;

    subplot(2,2,1);
    plot(1:ntrn, sugeno.onevsall.training_data(:,5), '*r', 1:ntrn, sugeno.onevsall.training_out(:,1), '*b');
    legend('Training Data', 'ANFIS Output');

    subplot(2,2,2);
    plot(1:nchk, sugeno.onevsall.validation_data(:,5), '*r', 1:nchk, sugeno.onevsall.validation_out(:,1), '*b');
    legend('Training Data', 'ANFIS Output');

    subplot(2,2,[3,4]);
    plot(1:epochs, sugeno.onevsall.train_err, '.r', 1:epochs, sugeno.onevsall.check_err, '*b');
    legend('Training error', 'Validation error');
end

