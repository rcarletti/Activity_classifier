function [features] = getfeatures(signal, domain)
%extracts features of the signal in frequency or time domain

    sampling_time = 0.082; %seconds
    sampling_frequency = 1/sampling_time;

    zci = @(v) find(v(:).*circshift(v(:),[-1,0])<=0);

    [ue,le] = envelope(signal); % Signal upper and lower envelopes
    [P,F] = periodogram(signal,[],length(signal),sampling_frequency,'power'); % Power Spectral Density
    PdBW = 10*log10(P); % PSD in decibels-watts
    [~,pi] = findpeaks(PdBW); % PSD peaks
    
    if strcmp(domain, 't')                  % time domain
        features = cell(1,6);
        features{1} = min(signal);          % Min
        features{2} = max(signal);          % Max
        features{3} = median(signal);       % Median
        features{4} = rms(signal);          % RMS
        features{5} = meanabs(signal);      % Mean of absolute values
        features{6} = sumabs(signal);       % Sum of absolute values
        features{7} = sumabs(diff(signal)); % Sum of absolute values of derivative
        features{8} = peak2rms(signal);     % Peak-magnitude-to-RMS ratio
        features{9} = peak2peak(signal);    % Maximum-to-minimum difference
        features{10} = rssq(signal);        % Root-Sum-of-Squares of the signal
        features{11} = length(zci(signal)); % Number of zero-crossing points
        features{12} = mean(ue);            % Mean of upper envelope
        features{13} = mean(le);            % Mean of lower envelope
    end

    if strcmp(domain, 'f')
        features = cell(1,4);
        features{1} = obw(signal,sampling_frequency);           % 99% occupied bandwidth
        [~,~,~,features{2}] = obw(signal, sampling_frequency);  % Power in the band 
        features{3} = meanfreq(signal);                         % Mean normalized frequency
        features{4} = medfreq(signal, sampling_frequency);      % Median frequency
        features{5} = bandpower(signal);                        % Average power
        features{6} = sum(afft(signal, sampling_time));         % Sum of spectral components
        features{7} = PdBW(F==0);                               % Power at DC
        features{8} = length(findpeaks(PdBW));                  % Number of peaks in PSD
        features{9} = mean(diff(F(pi)));                        % Average distance between peaks
        features{10} = sum(PdBW);                               % Total power
    end
end
