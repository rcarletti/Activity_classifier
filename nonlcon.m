function [c,ce] = nonlcon(x)
    ce = [];
    c = [-sum(x)+4];
end