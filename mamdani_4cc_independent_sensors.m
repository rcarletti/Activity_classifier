%% create mamdani fis

mamdani.fcc = newfis( 'mamdani-4cc-is',...
                      'FISType','mamdani',...
                      'AndMethod','min',...
                      'OrMethod','max',...
                      'DefuzzificationMethod','centroid',...
                      'ImplicationMethod', 'min', ...
                      'AggregationMethod', 'max');


%% add inputs to the FIS
% use the features of the best sensor previously computed (best_sensor_4cc)
% compute bounds for each feature

bounds = zeros(4,2);  %[min, max]
app = zeros(4,40);

for a_id = 1:4
    for v_id = 1:10
        for f_id = 1:4
            app(f_id,(a_id - 1) * 10 + v_id) = dsgetfeature(features_ds, best_sensor_4cc.features(f_id), best_sensor_4cc.index, a_id, v_id); 
        end
    end
end

for i = 1:4
    bounds(i,1) = min(app(i,:));
    bounds(i,2) = max(app(i,:));
    range = bounds(i,2) - bounds(i,1);
    bounds(i,1) = bounds(i,1) - range/10; 
end

for i = 1:4
    mamdani.fcc = addvar(mamdani.fcc,'input',features_names(best_sensor_4cc.features(i)), [bounds(i,1), bounds(i,2)]);
end

