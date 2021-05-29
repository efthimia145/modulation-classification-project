%% ======================================================================== %
%   Main function of constellations generation. 
%   This script will be used to generate 4 different modulations, using 
%   different SNR values, and different parameters for the Rayleigh
%   Channel. 
%   The final scatter plots will be used for the team to prepare the
%   dataset and, consequently, train a model for modulation classification
%   via DNN (or CNN).
%% ======================================================================== %

clear;
clc;

sampleRate = 5e5; % Sample rate
maxDopplerShift  = 50; % Maximum Doppler shift of diffuse components (Hz)
maxDopplerShift_null = 0;
delayVector = [0 90 210 440]*1e-9; % Discrete delays of four-path channel (s)
gainVector  = [0 -10.7 -18.1 -24.4]; % Average path gains (dB)

% Configure a Rayleigh channel object
rayChan = comm.RayleighChannel(...
    'SampleRate',sampleRate, ...
    'PathDelays',delayVector, ...
    'AveragePathGains',gainVector, ...
    'NormalizePathGains',true, ...
    'MaximumDopplerShift',maxDopplerShift_null, ...
    'DopplerSpectrum',{doppler('Flat'),doppler('Flat'),doppler('Flat'),doppler('Flat')}, ...
    'RandomStream','mt19937ar with seed', ...
    'Seed',22, ...
    'PathGainsOutputPort',true);

rayChan_doppler = comm.RayleighChannel(...
    'SampleRate',sampleRate, ...
    'PathDelays',delayVector, ...
    'AveragePathGains',gainVector, ...
    'NormalizePathGains',true, ...
    'MaximumDopplerShift',maxDopplerShift, ...
    'DopplerSpectrum',{doppler('Flat'),doppler('Flat'),doppler('Flat'),doppler('Flat')}, ...
    'RandomStream','mt19937ar with seed', ...
    'Seed',22, ...
    'PathGainsOutputPort',true);
    
NTrain = 5000;
NClass = 12;
L = 500;
SNR_low = 0;
SNR_high = 50;
SNR_step = 10;

count_classes = -1;


%% QAM 
tic
M_QAM = [4 8 16 64];
for M = M_QAM    
    count_classes = count_classes + 1;
    display = [num2str(count_classes), "-QAM"];
    disp(display)
    for SNR = SNR_low:SNR_step:SNR_high
        signal_data = zeros(NTrain, L);
        signal_label = zeros(NTrain, 1);
        
        display = ["SNR: ", num2str(SNR)];
        disp(display)
        for row = 1:NTrain
            
            bitsPerFrame_QAM = L;
            
            if unidrnd(2) == 2
                [ynoisy_qam] = QAM(M, bitsPerFrame_QAM, rayChan, SNR);
            else
                [ynoisy_qam] = QAM(M, bitsPerFrame_QAM, rayChan_doppler, SNR);
            end
            
            signal_data(row, :) = ynoisy_qam;
            signal_label(row, 1) = count_classes;
       
        end
        toc
        save(append('./dataset/train_data_', num2str(count_classes), '_', num2str(SNR),'.mat'), ...
                                                    'signal_data', '-mat');
        save(append('./dataset/train_label_', num2str(count_classes), '_', num2str(SNR),'.mat'), ...
                                                    'signal_label', '-mat');
        clearvars -except NTrain count_classes L rayChan rayChan_doppler ...
                            M_QAM M SNR SNR_low SNR_step SNR_high
                            
    end
end
disp("QAM is done")
c = clock;
c = fix(c);
fprintf("Files saved at: %d:%d:%d", c(4:5))


%% QPSK
tic
M = 4;
count_classes = count_classes + 1;
display = [num2str(count_classes), "-QPSK"];
disp(display)
for SNR = SNR_low:SNR_step:SNR_high
    signal_data = zeros(NTrain, L);
    signal_label = zeros(NTrain, 1);
    
    display = ["SNR: ", num2str(SNR)];
    disp(display)
    for row = 1:NTrain

        bitsPerFrame_QPSK = L;
        if unidrnd(2) == 2
            [ynoisy_qpsk] = QPSK(M, bitsPerFrame_QPSK, rayChan, SNR);
        else
            [ynoisy_qpsk] = QPSK(M, bitsPerFrame_QPSK, rayChan_doppler, SNR);
        end

        signal_data(row, :) = ynoisy_qpsk;
        signal_label(row, 1) = count_classes;
        
    end
    toc
    save(append('./dataset/train_data_', num2str(count_classes), '_', num2str(SNR),'.mat'), ...
                                                    'signal_data', '-mat');
    save(append('./dataset/train_label_', num2str(count_classes), '_', num2str(SNR),'.mat'), ...
                                                    'signal_label', '-mat');
    clearvars -except NTrain count_classes L rayChan rayChan_doppler ...
                        M SNR SNR_low SNR_step SNR_high
    
end
disp("QPSK is done")
c = clock;
c = fix(c);
fprintf("Files saved at: %d:%d:%d", c(4:5))

%% PAM
tic
M_PAM = [4 8 16 64];
for M = M_PAM
    count_classes = count_classes + 1;
    display = [num2str(count_classes), "-PAM"];
    disp(display)
    for SNR = SNR_low:SNR_step:SNR_high
        signal_data = zeros(NTrain, L);
        signal_label = zeros(NTrain, 1);
        
        display = ["SNR: ", num2str(SNR)];
        disp(display)
        for row = 1:NTrain
            
            bitsPerFrame_PAM = L;
            if unidrnd(2) == 2
                [ynoisy_pam] = PAM(M, bitsPerFrame_PAM, rayChan, SNR);
            else
                [ynoisy_pam] = PAM(M, bitsPerFrame_PAM, rayChan_doppler, SNR);
            end
            
            signal_data(row, :) = ynoisy_pam;
            signal_label(row, 1) = count_classes;
       
        end
        toc
        save(append('./dataset/train_data_', num2str(count_classes), '_', num2str(SNR),'.mat'), ...
                                                    'signal_data', '-mat');
        save(append('./dataset/train_label_', num2str(count_classes), '_', num2str(SNR),'.mat'), ...
                                                    'signal_label', '-mat');
        clearvars -except NTrain count_classes L rayChan rayChan_doppler ...
                            M_PAM M SNR SNR_low SNR_step SNR_high
    end
    
    
end
disp("PAM is done")
c = clock;
c = fix(c);
fprintf("Files saved at: %d:%d:%d", c(4:5))

%% APSK
tic
M_APSK = [4 8 20; 8 16 40; 16 32 80];
radiis = [0.3 0.7 1.2; 0.5 1 1.5; 1 2 3];
for k = 1:3
    M = M_APSK(k, :);
    radii = radiis(k, :);
    count_classes = count_classes + 1;
    display = [num2str(count_classes), "-APSK"];
    disp(display)
    for SNR = SNR_low:SNR_step:SNR_high
        signal_data = zeros(NTrain, L);
        signal_label = zeros(NTrain, 1);
        
        display = ["SNR: ", num2str(SNR)];
        disp(display)
        for row = 1:NTrain
            
            bitsPerFrame_APSK = L;
            if unidrnd(2) == 2
                [ynoisy_apsk] = APSK(M, radii, bitsPerFrame_APSK, rayChan, SNR);
            else
                [ynoisy_apsk] = APSK(M, radii, bitsPerFrame_APSK, rayChan_doppler, SNR);
            end
            
            
            signal_data(row, :) = ynoisy_apsk;
            signal_label(row, 1) = count_classes;
       
        end
        toc
        save(append('./dataset/train_data_', num2str(count_classes), '_', num2str(SNR),'.mat'), ...
                                                    'signal_data', '-mat');
        save(append('./dataset/train_label_', num2str(count_classes), '_', num2str(SNR),'.mat'), ...
                                                    'signal_label', '-mat');
        clearvars -except NTrain count_classes L rayChan rayChan_doppler ...
                            M_APSK M SNR SNR_low SNR_step SNR_high radiis radii k
    end
end
disp("APSK is done")
c = clock;
c = fix(c);
fprintf("All Files ready at: %d:%d:%d", c(4:5))

%% Constellations Diagramms
% constDiag_qam(ynoisy_qam)
% constDiag_pam(ynoisy_pam)
% constDiag_qpsk(ynoisy_qpsk)
% constDiag_apsk(ynoisy_apsk)
        
