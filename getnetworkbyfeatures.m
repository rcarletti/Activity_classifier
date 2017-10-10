function [nn] = getnetworkbyfeatures(features, sensor)
    global neural_networks
    global C

    for i = 1:size(C, 1)
        if isequal(C(i,:),features)
            nn = neural_networks{sensor}{i};
            break;
        end
    end
end

