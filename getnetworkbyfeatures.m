function [network, index] = getnetworkbyfeatures(features, sensor)
%UNTITLED3 Summary of this function goes here
    global neural_networks_1
    global neural_networks_2
    global neural_networks_3
    global C
    index = 0;
    for i=1:length(C)
        if isequal(C(i,:),features)
            index = i;
            break;
        end
    end
    if sensor == 1
        network = neural_networks_1{index};
    elseif sensor == 2
        network = neural_networks_2{index};
    else
        network = neural_networks_3{index};
    end
end

