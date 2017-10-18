%% choose parameters for ANFIS - four class classifier (independent sensors)

%create the matrix with features for each couple activity-volunteer
%the data are arranged in [feat1,feat2,feat3,feat4,activity,volunteer]
    
a_v_matrix = zeros(40,6);
for a_id = 1:4
    for v_id = 1:10
        for f_id = 1:4
            a_v_matrix((a_id -1) *10 + v_id, f_id) = dsgetfeature(features_ds,...
                mod(best_all_4cc.features(f_id),11) + 1, ...
                ceil(best_all_4cc.features(f_id)/11),...
                a_id, ...
                v_id);
        end
        a_v_matrix((a_id -1) * 10 + v_id, 5) = a_id;
        a_v_matrix((a_id -1) * 10 + v_id, 6) = v_id;
    end
end

%shuffle columns
a_v_matrix_perm = a_v_matrix(randperm(size(a_v_matrix,1)),:);

% select ANFIS data - 70%-30% split

ntrn = floor(40*0.7);
nchk = 40-ntrn;

sugeno.fcc_all.training_data = a_v_matrix_perm(1:ntrn, 1:5);
sugeno.fcc_all.validation_data = a_v_matrix_perm(ntrn+1:40, 1:5);

%% generate and train the sugeno FIS

nmfs = 2;       %membership functions
epochs = 150;

% generate initial FIS
sugeno.fcc_all.genopt = genfisOptions('GridPartition');
sugeno.fcc_all.genopt.NumMembershipFunctions = nmfs;
sugeno.fcc_all.genopt.InputMembershipFunctionType = 'gbellmf';


% set FIS options
sugeno.fcc_all.fisopt = anfisOptions('EpochNumber', epochs, 'OptimizationMethod', 1, 'InitialFIS', ...
    genfis(sugeno.fcc.training_data(:,1:4), sugeno.fcc.training_data(:,5), sugeno.fcc.genopt));
sugeno.fcc_all.fisopt.DisplayErrorValues = 0;
sugeno.fcc_all.fisopt.DisplayStepSize = 0;
sugeno.fcc_all.fisopt.ValidationData = sugeno.fcc.validation_data;

% run ANFIS
[sugeno.fcc_all.fis, sugeno.fcc_all.train_err, ~, sugeno.fcc_all.check_fis, sugeno.fcc_all.check_err] = ...
    anfis(sugeno.fcc_all.training_data, sugeno.fcc_all.fisopt);


% compute fuzzy output values
sugeno.fcc_all.training_out = evalfis(sugeno.fcc_all.training_data(:,1:4), sugeno.fcc_all.fis);
sugeno.fcc_all.validation_out = evalfis(sugeno.fcc_all.validation_data(:,1:4), sugeno.fcc_all.fis);

% plot output data
figure;

subplot(2,2,1);
plot(1:ntrn, sugeno.fcc_all.training_data(:,5), '*r', 1:ntrn, sugeno.fcc_all.training_out(:,1), '*b');
legend('Training Data', 'ANFIS Output');

subplot(2,2,2);
plot(1:nchk, sugeno.fcc_all.validation_data(:,5), '*r', 1:nchk, sugeno.fcc_all.validation_out(:,1), '*b');
legend('Training Data', 'ANFIS Output');

subplot(2,2,[3,4]);
plot(1:epochs, sugeno.fcc_all.train_err, '.r', 1:epochs, sugeno.fcc_all.check_err, '*b');
legend('Training error', 'Validation error');

