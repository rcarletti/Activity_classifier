%% main
load('data.mat');

filtered_s = dsnew();

res{1} = cell(4,10);
res{2} = cell(4,10);
res{3} = cell(4,10);


%% pre-process data

%filtering
for s_id = 1:3                          %for each sensor
    for a_id = 1:4                      %for each actovity
        for v_id  = 1:10                %for each volunteer
            fs = sgolayfilt(dsget(data,s_id,a_id,v_id),4,21);
            filtered_s = dsput(filtered_s,fs,s_id,a_id,v_id);
        end
    end
end

plot(dsget(data,1,1,1));
hold on 
plot(dsget(filtered_s,1,1,1));

%% bla
%subtract mean value

for s_id = 1:3
    for a_id = 1:4
        for v_id = 1:10
            ds = detrend(dsget(filtered_s, s_id, a_id, v_id),'constant');
            filtered_s = dsput(filtered_s,ds, s_id,a_id,v_id);
        end
    end
end

plot(dsget(filtered_s,1,1,1));


%delete outliers(?)
% for s_id = 1:3
%     for a_id = 1:4
%         for v_id = 1:10
%             %sensor i, activity j, volunteer k
%             res{s_id}{a_id,v_id}(:,1)= isoutlier(filtered_s{s_id}{a_id,v_id}(:,1))
%         end
%     end
% end

%plot(res{1}{1,1}(:,1) * 1.3 * (10^4))

%normalization

