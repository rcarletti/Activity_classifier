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

    sugeno.fcc.training_data = a_v_matrix_perm(1:ntrn, 1:5);
    sugeno.fcc.validation_data = a_v_matrix_perm(ntrn+1:40, 1:5);

    % generate and train the sugeno FIS

    nmfs = 2;
    epochs = 150;

    % generate initial FIS
    sugeno.fcc.genopt = genfisOptions('GridPartition');
    sugeno.fcc.genopt.NumMembershipFunctions = nmfs;
    sugeno.fcc.genopt.InputMembershipFunctionType = 'gbellmf';

    % set FIS options
    sugeno.fcc.fisopt = anfisOptions('EpochNumber', epochs, 'OptimizationMethod', 1, 'InitialFIS', ...
        genfis(sugeno.fcc.training_data(:,1:4), sugeno.fcc.training_data(:,5), sugeno.fcc.genopt));
    sugeno.fcc.fisopt.DisplayErrorValues = 0;
    sugeno.fcc.fisopt.DisplayStepSize = 0;
    sugeno.fcc.fisopt.ValidationData = sugeno.fcc.validation_data;

    % run ANFIS
    [sugeno.fcc.fis, sugeno.fcc.train_err, ~, sugeno.fcc.check_fis, sugeno.fcc.check_err] = ...
        anfis(sugeno.fcc.training_data, sugeno.fcc.fisopt);

    % compute fuzzy output values
    sugeno.fcc.training_out = evalfis(sugeno.fcc.training_data(:,1:4), sugeno.fcc.fis);
    sugeno.fcc.validation_out = evalfis(sugeno.fcc.validation_data(:,1:4), sugeno.fcc.fis);

    % plot output data
    figure;

    subplot(2,2,1);
    plot(1:ntrn, sugeno.fcc.training_data(:,5), '*r', 1:ntrn, sugeno.fcc.training_out(:,1), '*b');
    legend('Training Data', 'ANFIS Output');

    subplot(2,2,2);
    plot(1:nchk, sugeno.fcc.validation_data(:,5), '*r', 1:nchk, sugeno.fcc.validation_out(:,1), '*b');
    legend('Training Data', 'ANFIS Output');

    subplot(2,2,[3,4]);
    plot(1:epochs, sugeno.fcc.train_err, '.r', 1:epochs, sugeno.fcc.check_err, '*b');
    legend('Training error', 'Validation error');
end

