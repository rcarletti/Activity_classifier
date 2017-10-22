function [features] = getfeatures(signal, domain)
%extracts features of the signal in frequency or time domain

    sampling_time = 0.082; %seconds
    sampling_frequency = 1/sampling_time;

    if strcmp(domain, 't')                  %time domain
        features = cell(1,7);
        features{1} = min(signal);          %min
        features{2} = max(signal);          %max
        features{3} = mean(signal);         %mean
        features{4} = std(signal);          %standard deviation
        features{5} = peak2rms(signal);     %Peak-magnitude-to-RMS ratio
        features{6} = peak2peak(signal);    %Maximum-to-minimum difference
        features{7} = rssq(signal);         %Root-Sum-of-Squares of the signal
    end

    if strcmp(domain, 'f')
        features = cell(1,4);
        features{1} = obw(signal,sampling_frequency);           %99% occupied bandwidth
        [~,~,~,features{2}] = obw(signal, sampling_frequency);  % power in the band 
        features{3} = meanfreq(signal); %mean normalized frequency of the power spectrum
        features{4} = bandpower(signal); %average power
    end
end
