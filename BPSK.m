function [ynoisy_bpsk, constDiag_bpsk] = BPSK(M, bitsPerFrame_BPSK, rayChan, SNR)
% function [ynoisy_bpsk] = BPSK(M, bitsPerFrame_BPSK, rayChan, SNR)

    %% BPSK

    x = randi([0 M-1],bitsPerFrame_BPSK,1); % Input signal

    bpskmod = comm.BPSKModulator;
    bpskdemod = comm.BPSKDemodulator;
    cpts = bpskmod(x);

    constDiag_bpsk = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
        'XLimits',[-M M],'YLimits',[-M M]);

    y = bpskmod(x);
    ynoisy_bpsk = awgn(y,SNR,'measured'); % Noise addition (SNR)

    ynoisy_bpsk = rayChan(ynoisy_bpsk);

    z = bpskdemod(ynoisy_qpsk);
    [num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate

    fprintf("Number of symbol errors: %f\n", num);
    fprintf("Symbol error rate: %f\n", rt);
    
end