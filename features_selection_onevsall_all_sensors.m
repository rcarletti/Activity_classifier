
global onevsall_all;
for i = 1:4
    onevsall_all{i} = struct;
end

%% retrieve best features and sensor for each classifier


    retrievebestfeatures(i,features_ds);


function [] = retrievebestfeatures(act, features_ds)
    
    global total_features;
    global chosen_features_num;
    global onevsall_all;
    
    %generate targets for class act
   
    targets = zeros(2, 120);
    targets(2,:) = ones(1, 120);
    targets(1, (1:30) + (act-1)*10) = ones(1,30);
    targets(2, (1:30) + (act-1)*10) = zeros(1,30);

    
    %set up the GA
    %this time we consider 33 features, 11 features for each sensor
    population_size = 100;
    population_all = zeros(100, total_features * 3);

    %generate random population (pop_size x total_features array, features set to 1 are
    %the chosen features for that individual
    for i=1:population_size
        feat_perm = randperm(total_features * 3, chosen_features_num);
        for f_id =1:(chosen_features_num)
            population_all(i,feat_perm(f_id)) = 1;
        end
    end

    options = gaoptimset(@ga);
    options.PopulationType = 'doubleVector';
    options.InitialPopulation = population_all;
    options.useParallel = 'true';

    intcon = (1:total_features * 3);
    nonlinearcon = @(x)nonlcon(x);

    %run the genetic algoritm

    for i=1:4
        feats = ga(@(x) fitnessall(x,...
                targets,...
                features_ds, ...
                strcat(int2str(i),'vsall')), ...
                total_features * 3,...
                [], [], [], [], ...
                zeros(1,total_features * 3), ones(1,total_features * 3), ...
                nonlinearcon, ...
                intcon, ...
                options);
    disp(strcat('-------end of class n.', num2str(i)));
    end
    
    onevsall_all{act}.features = genes2feat(feats);

end
