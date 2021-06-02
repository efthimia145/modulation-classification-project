function [ynoisy_pam, constDiag_pam] = PAM(M, bitsPerFrame_PAM, rayChan, SNR)
% function [ynoisy_pam] = PAM(M, bitsPerFrame_PAM, rayChan, SNR)

    %% PAM

    x = randi([0 M-1],bitsPerFrame_PAM,1); % Input signal

%     cpts = pammod(x,M,pi/4);
%     constDiag_pam = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
%         'XLimits',[-M M],'YLimits',[-M M]);

    y = pammod(x,M,pi/4);
    ynoisy_pam = awgn(y,SNR,'measured'); % Noise addition (SNR)

    % release(rayChan)
    ynoisy_pam = rayChan(ynoisy_pam);

%     z = pamdemod(ynoisy_pam,M,pi/4);
%     [num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate
% 
%     fprintf("Number of symbol errors: %f\n", num);
%     fprintf("Symbol error rate: %f\n", rt);

end