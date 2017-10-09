function [features] = getfeatures(signal, domain)
%extracts features of the signal in frequency or time domain
    features = cell(1,8);
    if strcmp(domain, 't')                  %time domain
        features{1} = min(signal);          %min
        features{2} = max(signal);          %max
        features{3} = mean(signal);         %mean
        features{4} = std(signal);          %standard deviation
        features{5} = peak2rms(signal);     %Peak-magnitude-to-RMS ratio
        features{6} = peak2peak(signal);    %Maximum-to-minimum difference
        features{7} = rssq(signal);         %Root-Sum-of-Squares of the signal
        features{8} = envelope(signal);     %Signal envelope
    end
    
    if strcmp(domain, 'f')
    end
    
end

