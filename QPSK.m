function [ynoisy_qpsk, constDiag_qpsk] = QPSK(M, bitsPerFrame_QPSK, rayChan, SNR)
% function [ynoisy_qpsk] = QPSK(M, bitsPerFrame_QPSK, rayChan, SNR)

    %% QPSK

    x = randi([0 M-1],bitsPerFrame_QPSK,1); % Input signal

    qpskmod = comm.QPSKModulator;
%     qpskdemod = comm.QPSKDemodulator;
%     cpts = qpskmod(x);
% 
%     constDiag_qpsk = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
%         'XLimits',[-M M],'YLimits',[-M M]);

    y = qpskmod(x);
    ynoisy_qpsk = awgn(y,SNR,'measured'); % Noise addition (SNR)

    % release(rayChan)
    ynoisy_qpsk = rayChan(ynoisy_qpsk);

%     z = qpskdemod(ynoisy_qpsk);
%     [num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate
% 
%     fprintf("Number of symbol errors: %f\n", num);
%     fprintf("Symbol error rate: %f\n", rt);
    
end
