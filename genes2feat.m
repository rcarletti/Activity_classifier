function [f] = genes2feat(g)
% Convert features expresses as GA genes into a feature array
    j = 1;
    f = zeros(1,4);
    for i = 1:length(g)
        if g(i) == 1
            f(j) = i;
            j = j+1;
        end
    end
end
