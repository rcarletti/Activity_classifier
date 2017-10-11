function [nn] = getnetworkbyfeatures(features, sensor)
    global neural_networks
    global C
    global total_features
    
    if(length(features)==11)
        j = 1;
        app = zeros(1,4);
        for i=1:total_features
            if features(i) == 1
                app(j) = i;
                j = j+1;
            end
        end
        features = app;
    end

    for i = 1:size(C, 1)
        if isequal(C(i,:),features)
            nn = neural_networks{sensor}{i};
            break;
        end
    end
end

