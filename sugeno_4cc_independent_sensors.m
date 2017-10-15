%% choose parameters for ANFIS - four class classifier (independent sensors)

%create the matrix with features for each couple activity-volunteer
%the data are arranged in [feat1,feat2,feat3,feat4,activity,volunteer]
    
a_v_matrix = zeros(40,6);
for a_id = 1:4
    for v_id = 1:10
        for f_id = 1:4
            a_v_matrix((a_id -1) *10 + v_id, f_id) = dsgetfeature(features_ds,...
                best_sensor_4cc.features(f_id), ...
                best_sensor_4cc.index, ...
                a_id, ...
                v_id);
        end
        a_v_matrix((a_id -1) * 10 + v_id, 5) = a_id;
        a_v_matrix((a_id -1) * 10 + v_id, 6) = v_id;
    end
end

%shuffle columns

a_v_matrix_perm = a_v_matrix(randperm(size(a_v_matrix,1)),:);

%select checking data - 15% of the dataset - 6 rows of the matrix

sugeno.fcc.checking_data = a_v_matrix_perm(1:6, 1:5);

%select testing data - 15% of the dataset - 6 rows of the matrix

sugeno.fcc.testing_data = a_v_matrix_perm(7:12, 1:5);

%select training data - 70% of the dataset - 28 rows of the matrix

sugeno.fcc.training_data = a_v_matrix_perm(13:40, 1:5);

%% generate the sugeno FIS

