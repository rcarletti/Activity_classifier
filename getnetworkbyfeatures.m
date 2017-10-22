function [nn] = getnetworkbyfeatures(nets, features)
    for i = 1:length(nets)
        if isequal(nets{i}.features, features)
            nn = nets{i};
            break;
        end
    end
end
