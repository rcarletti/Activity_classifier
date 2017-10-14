function [nn] = getnetworkbyfeatures(features, sensor, nn_type)
    global neural_networks
    global neural_networks_1vsall;
    global neural_networks_2vsall;
    global neural_networks_3vsall;
    global neural_networks_4vsall;
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
            if strcmp(nn_type,'4cc')
                    nn = neural_networks{sensor}{i};
                elseif strcmp(nn_type,'1vsall')
                    nn = neural_networks_1vsall{sensor}{i};
                elseif strcmp(nn_type,'2vsall')
                    nn = neural_networks_2vsall{sensor}{i};
                elseif strcmp(nn_type,'3vsall')
                    nn = neural_networks_3vsall{sensor}{i};
                else 
                    nn = neural_networks_4vsall{sensor}{i};
            end
            break;
        end
    end
end

