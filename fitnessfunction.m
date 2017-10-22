function [conf] = fitnessfunction(nets, feature_set)
%retrieves the nn trained with the features specified in the feature set
%and computes the confusion associated to it.
    nn = getnetworkbyfeatures(nets, genes2feat(feature_set));
    conf = nn.conf;
end
