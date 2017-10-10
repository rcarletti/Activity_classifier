function [nns] = createandtrainnn(sensor_num,inputs,targets, nets_num)
%creates and trains nets_num neural networks with inputs and targets from sensor
%sensor_num
    global C

    train_iter = 5;
    nns = cell(1,nets_num);

    for i=1:nets_num
        nns{i} = {};
        
        nns{i}.net = patternnet(10);
        nns{i}.net.divideParam.trainRatio = 70/100;
        nns{i}.net.divideParam.valRatio = 15/100;
        nns{i}.net.divideParam.testRatio = 15/100;

        nns{i}.inputs = inputs{sensor_num}(:,:,i);
        nns{i}.targets = targets;
        nns{i}.features = C(i,:);
        
        for j=1:train_iter
            nns{i}.net = train(nns{i}.net, ...
                nns{i}.inputs, nns{i}.targets);
        end

        nns{i}.results = nns{i}.net(nns{i}.inputs);
        nns{i}.perf = perform(nns{i}.net, nns{i}.targets, nns{i}.results);
        nns{i}.conf = confusion(nns{i}.targets, nns{i}.results);
    end
end
