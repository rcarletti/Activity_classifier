function [neural_networks] = createandtrainnn(sensor_num,inputs,targets, nets_num)
%creates and trains nets_num neural networks with inputs and targets from sensor
%sensor_num
    neural_networks = cell(1,nets_num);
    results_nn = cell(1,nets_num);
    performance_nn = cell(1,nets_num);
    for i=1:nets_num
        neural_networks{i} = patternnet(10);
        neural_networks{i}.divideParam.trainRatio = 70/100;
        neural_networks{i}.divideParam.valRatio = 15/100;
        neural_networks{i}.divideParam.testRatio = 15/100;
        for j=1:5
            %train 5 times
            neural_networks{i} = train(neural_networks{i},inputs{sensor_num}(:,:,1), targets(:,:,1));
        end
        %results_nn{i} = neural_networks{i}(inputs{sensor_num}(:,:,1));
        %performance_nn{i} = perform(neural_networks{i},targets(:,:,1),results_nn{i});
    end

end

