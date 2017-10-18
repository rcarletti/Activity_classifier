%% ANFIS one-vs-all

seed = floor(rand * 10000);

sugeno_1vsall.onevsall_all = anfis_onevsall_all(best_onevsall_all{1,1}, 1, features_ds, seed);
sugeno_2vsall.onevsall_all = anfis_onevsall_all(best_onevsall_all{1,2}, 2, features_ds, seed);
sugeno_3vsall.onevsall_all = anfis_onevsall_all(best_onevsall_all{1,3}, 3, features_ds, seed);
sugeno_4vsall.onevsall_all = anfis_onevsall_all(best_onevsall_all{1,4}, 4, features_ds, seed);

%% Local functions

function [onevsall_all] = anfis_onevsall_all(best_sensor, act, features_ds, seed)
% choose parameters for ANFIS - one-vs-all classifier (independent sensors)

    %create the matrix with features for each couple activity-volunteer
    %the data are arranged in [feat1,feat2,feat3,feat4,activity,volunteer]

    a_v_matrix = zeros(40,6);
    for a_id = 1:4
        for v_id = 1:10
            for f_id = 1:4
                a_v_matrix((a_id -1) *10 + v_id, f_id) = dsgetfeature(features_ds,...
                    mod(best_sensor.features(f_id),11) + 1, ...
                    ceil(best_sensor.features(f_id)/11),...
                    a_id, ...
                    v_id);
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

    onevsall_all.training_data = a_v_matrix_perm(1:ntrn, 1:5);
    onevsall_all.validation_data = a_v_matrix_perm(ntrn+1:40, 1:5);

    % generate and train the sugeno FIS

    nmfs = 2;
    epochs = 150;

    % generate initial FIS
    onevsall_all.genopt = genfisOptions('GridPartition');
    onevsall_all.genopt.NumMembershipFunctions = nmfs;
    onevsall_all.genopt.InputMembershipFunctionType = 'gbellmf';

    % set FIS options
    onevsall_all.fisopt = anfisOptions('EpochNumber', epochs, 'OptimizationMethod', 1, 'InitialFIS', ...
        genfis(onevsall_all.training_data(:,1:4), onevsall_all.training_data(:,5), onevsall_all.genopt));
    onevsall_all.fisopt.DisplayErrorValues = 0;
    onevsall_all.fisopt.DisplayStepSize = 0;
    onevsall_all.fisopt.ValidationData = onevsall_all.validation_data;

    % run ANFIS
    [onevsall_all.fis, onevsall_all.train_err, ~, onevsall_all.check_fis, onevsall_all.check_err] = ...
        anfis(onevsall_all.training_data, onevsall_all.fisopt);

    % compute fuzzy output values
    onevsall_all.training_out = evalfis(onevsall_all.training_data(:,1:4), onevsall_all.fis);
    onevsall_all.validation_out = evalfis(onevsall_all.validation_data(:,1:4), onevsall_all.fis);

    % plot output data
    figure;

    subplot(2,2,1);
    plot(1:ntrn, onevsall_all.training_data(:,5), '*r', 1:ntrn, onevsall_all.training_out(:,1), '*b');
    legend('Training Data', 'ANFIS Output');

    subplot(2,2,2);
    plot(1:nchk, onevsall_all.validation_data(:,5), '*r', 1:nchk, onevsall_all.validation_out(:,1), '*b');
    legend('Training Data', 'ANFIS Output');

    subplot(2,2,[3,4]);
    plot(1:epochs, onevsall_all.train_err, '.r', 1:epochs, onevsall_all.check_err, '*b');
    legend('Training error', 'Validation error');
end


    
    
