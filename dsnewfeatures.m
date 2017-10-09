function [dataset] = dsnewfeatures()
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    dataset = cell(4,10);
    for s = 1:3
        for a=1:4
            for v=1:10
                dataset{a,v}{s} = cell(1,8);
            end
        end
    end
    
end

