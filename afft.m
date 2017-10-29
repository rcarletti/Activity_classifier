function [Y,f] = afft(x, Ts)
% FFT of signal x sampled with period Ts
    L = length(x);
    Fs = 1/Ts;
    
    Y = abs(fft(x)/L);
    Y = Y(1:L/2+1);
    Y(2:end-1) = 2*Y(2:end-1);
    f = Fs*(0:L/2)/L;
end

