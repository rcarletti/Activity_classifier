%% create mamdani fis

for time_interval = [1,2,4]
    [fcc_ind{time_interval}.mamdani, feat_mat] = perform_fis(...
        fcc_ind{time_interval}.best_sensor, fcc_ind{time_interval}.fis_input, ...
        fcc_ind{time_interval}.sugeno.fis, features_ds, features_names, time_interval);
end

function [mamdani, app] = perform_fis(sensor, input, sugeno, features_ds, features_names, time_interval)

    % create Mamdani FIS

    mamdani.fis = newfis('mamdani-4cc-ind', ...
                         'FISType',               'mamdani',...
                         'AndMethod',             'min',...
                         'OrMethod',              'max',...
                         'DefuzzificationMethod', 'centroid',...
                         'ImplicationMethod',     'min', ...
                         'AggregationMethod',     'max');

    mamdani.input = input;
                     
    % add inputs to the FIS using the features of the best sensor previously computed

    bounds = zeros(4,2);  %[min, max]
    app = zeros(4,40 * time_interval);

    for a_id = 1:4
        for v_id = 1:(10 * time_interval)
            for f_id = 1:4
                app(f_id, (a_id - 1) * 10 * time_interval + v_id) = ...
                    dsgetfeature(features_ds, sensor.features(f_id), ...
                    sensor.index, a_id, v_id, time_interval); 
            end
        end
    end

    % compute bounds for each feature

    for i = 1:4
        % account for a possible 10% inaccuracy
        bounds(i,1) = min(app(i,:));
        bounds(i,2) = max(app(i,:));
        range = bounds(i,2) - bounds(i,1);
        bounds(i,1) = bounds(i,1) - range * 0.1; 
        bounds(i,2) = bounds(i,2) + range * 0.1; 

        % add variable to the Mamdani network
        mamdani.fis = addvar(mamdani.fis, 'input', ...
            features_names(sensor.features(i)), ...
            [bounds(i,1), bounds(i,2)]);
    end

    % add stuff from sugeno
    
    for i = 1:4
        mamdani.fis.input(i).mf = sugeno.input(i).mf;
    end
    mamdani.fis.rule = sugeno.rule;
    mamdani.fis.output = sugeno.output;

    % compute fuzzy output values

    %mamdani.eval_out = evalfis(input(:,1:4), mamdani.fis);
    
    %plot(1:40, mamdani.input(:,5), '*r', 1:40, mamdani.eval_out(:,1), '*b');

end
