function [ynoisy_qam, constDiag_qam] = QAM(M, bitsPerFrame_QAM, rayChan, SNR)
% function [ynoisy_qam] = QAM(M, bitsPerFrame_QAM, rayChan, SNR)

    %% QAM 
    
    x = randi([0 M-1],bitsPerFrame_QAM,1); % Input signal

%     cpts = qammod(0:M-1,M);
%     constDiag_qam = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
%         'XLimits',[-M M],'YLimits',[-M M]);

    y = qammod(x,M);
    ynoisy_qam = awgn(y,SNR,'measured'); % Noise addition (SNR)

    % release(rayChan)
    ynoisy_qam = rayChan(ynoisy_qam);

%     z = qamdemod(ynoisy_qam,M);
%     [num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate
% 
%     fprintf("Number of symbol errors: %f\n", num);
%     fprintf("Symbol error rate: %f\n", rt);

end
