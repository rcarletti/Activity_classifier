function [y] = fitnessfunction(nets_num,neural_networks,sensor_num,inputs,targets)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    results_nn = cell(1,nets_num);
    performance_nn = cell(1,nets_num);
    for i=1:net_nums
        results_nn{i} = neural_networks{i}(inputs{sensor_num}(:,:,1));
        performance_nn{i} = perform(neural_networks{i},targets(:,:,1),results_nn{i});
    end
    y = min(performance_nn);
end

