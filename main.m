load('data.mat');

filtered_s = dsnew();
normalized_s = dsnew();
chosen_features_num = 4;

global total_features;

%% filter data

for s_id = 1:3                          %for each sensor
    for a_id = 1:4                      %for each actovity
        for v_id  = 1:10                %for each volunteer
            fs = sgolayfilt(dsget(data,s_id,a_id,v_id),4,21);
            filtered_s = dsput(filtered_s,fs,s_id,a_id,v_id);
        end
    end
end

%plot(dsget(data,1,1,1));
%hold on 
%plot(dsget(filtered_s,1,1,1));

%% subtract mean value

for s_id = 1:3
    for a_id = 1:4
        for v_id = 1:10
            ds = detrend(dsget(filtered_s, s_id, a_id, v_id),'constant');
            filtered_s = dsput(filtered_s,ds, s_id,a_id,v_id);
        end
    end
end

%plot(dsget(filtered_s,1,1,1));


%% z-normalization

for s_id = 1:3
    for a_id = 1:4
        for v_id = 1:10
            s = dsget(filtered_s,s_id,a_id,v_id);
            %compute standard deviation for each signal
            standard_dev = std(s);
            %divide the signal for the standard deviation
            normalized_signal = s/standard_dev;
            normalized_s = dsput(normalized_s, normalized_signal, s_id,...
                a_id, v_id);     
        end
    end
end

%plot(dsget(normalized_s,1,1,1));

%% extract features for each sensor 
features_ds = dsnew();

for s_id = 1:3
    for a_id = 1:4
        for v_id = 1:10
            feat_t = getfeatures(dsget(normalized_s,s_id,a_id,v_id), 't');
            feat_f = getfeatures(dsget(normalized_s,s_id,a_id,v_id), 'f');
            feat = [feat_t, feat_f];
            total_features = length(feat);
            features_ds = dsputfeatures(features_ds,feat,s_id,a_id,v_id);
        end
    end
end

%% features selection for the 4 class classifier (independent sensors)

features_selection_4cc_independent_sensors

%% features selection for the one-against-all classifier (independent sensors)

features_selection_onevsall_independent_sensors