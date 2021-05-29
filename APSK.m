% function [ynoisy_apsk, constDiag_apsk] = APSK(M, radii, bitsPerFrame_APSK, rayChan, SNR)
function [ynoisy_apsk] = APSK(M, radii, bitsPerFrame_APSK, rayChan, SNR)

    %% APSK 

    modOrder = sum(M);
    x = randi([0 modOrder-1],bitsPerFrame_APSK,1); % Input signal

%     cpts = apskmod(x,M,radii);
%     constDiag_apsk = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
%         'XLimits',[-2*radii(3) 2*radii(3)],'YLimits',[-2*radii(3) 2*radii(3)]);

    y = apskmod(x,M,radii);
    ynoisy_apsk = awgn(y,SNR,'measured'); % Noise addition (SNR)

    % release(rayChan)
    ynoisy_apsk = rayChan(ynoisy_apsk);

%     z = apskdemod(ynoisy_apsk,M,radii);
%     [num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate
% 
% 
%     fprintf("Number of symbol errors: %f\n", num);
%     fprintf("Symbol error rate: %f\n", rt);
    
end