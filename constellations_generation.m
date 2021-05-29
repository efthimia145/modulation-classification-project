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
    
NTrain = 2000;
NClass = 12;
L = 500;
SNR_low = 0;
SNR_high = 50;
SNR_step = 10;
train_data = zeros(NClass * NTrain, L);
train_label = zeros(NClass * NTrain, 1);
signal_data = zeros(NTrain, L);
count_classes = -1;
counter_N = -1;

tic

%% QAM 
M_QAM = [4 8 16 64];
for M = M_QAM
    count_classes = count_classes + 1;
    display = [num2str(count_classes), "-QAM"];
    disp(display)
    for SNR = SNR_low:SNR_step:SNR_high
        disp(SNR)
        counter_N = counter_N + 1;
        for row = 1:NTrain
            
            bitsPerFrame_QAM = L;
            
            if unidrnd(2) == 2
                [ynoisy_qam, constDiag_qam] = QAM(M, bitsPerFrame_QAM, rayChan, SNR);
            else
                [ynoisy_qam, constDiag_qam] = QAM(M, bitsPerFrame_QAM, rayChan_doppler, SNR);
            end
            
            train_data(row + counter_N*NTrain, :) = ynoisy_qam;
            train_label(row + counter_N*NTrain, 1) = count_classes;
       
        end
        toc
    end
end
disp("QAM is done")
save('./dataset/train_data_QAM.mat', 'train_data', '-mat');
save('./dataset/train_label_QAM.mat', 'train_label', '-mat');
disp("Files saved")

%% QPSK
count_classes = count_classes + 1;
display = [num2str(count_classes), "-QPSK"];
disp(display)
for SNR = SNR_low:SNR_step:SNR_high
    disp(SNR)
    counter_N = counter_N + 1;
    for row = 1:NTrain

        bitsPerFrame_QPSK = L;
        if unidrnd(2) == 2
            [ynoisy_qpsk, constDiag_qpsk] = QPSK(M, bitsPerFrame_QPSK, rayChan, SNR);
        else
            [ynoisy_qpsk, constDiag_qpsk] = QPSK(M, bitsPerFrame_QPSK, rayChan_doppler, SNR);
        end

        train_data(row + counter_N*NTrain, :) = ynoisy_qpsk;
        train_label(row + counter_N*NTrain, 1) = count_classes;

    end
    toc
end
disp("QPSK is done")
save('./dataset/train_data_QPSK.mat', 'train_data', '-mat');
save('./dataset/train_label_QPSK.mat', 'train_label', '-mat');
disp("Files saved")

%% PAM 
M_PAM = [4 8 16 64];
for M = M_PAM
    count_classes = count_classes + 1;
    display = [num2str(count_classes), "-PAM"];
    disp(display)
    for SNR = SNR_low:SNR_step:SNR_high
        disp(SNR)
        counter_N = counter_N + 1;
        for row = 1:NTrain
            
            bitsPerFrame_PAM = L;
            if unidrnd(2) == 2
                [ynoisy_pam, constDiag_pam] = PAM(M, bitsPerFrame_PAM, rayChan, SNR);
            else
                [ynoisy_pam, constDiag_pam] = PAM(M, bitsPerFrame_PAM, rayChan_doppler, SNR);
            end
            
            
            train_data(row + counter_N*NTrain, :) = ynoisy_pam;
            train_label(row + counter_N*NTrain, 1) = count_classes;
       
        end
    end
    toc
end
disp("PAM is done")
save('./dataset/train_data_PAM.mat', 'train_data', '-mat');
save('./dataset/train_label_PAM.mat', 'train_label', '-mat');
disp("Files saved")

%% APSK
M_APSK = [4 8 20; 8 16 40; 16 32 80];
radiis = [0.3 0.7 1.2; 0.5 1 1.5; 1 2 3];
for k = 1:3
    M = M_APSK(k, :);
    radii = radiis(k, :);
    count_classes = count_classes + 1;
    display = [num2str(count_classes), "-APSK"];
    disp(display)
    for SNR = SNR_low:SNR_step:SNR_high
        disp(SNR)
        counter_N = counter_N + 1;
        for row = 1:NTrain
            
            bitsPerFrame_APSK = L;
            if unidrnd(2) == 2
                [ynoisy_apsk, constDiag_apsk] = APSK(M, radii, bitsPerFrame_APSK, rayChan, SNR);
            else
                [ynoisy_apsk, constDiag_apsk] = APSK(M, radii, bitsPerFrame_APSK, rayChan_doppler, SNR);
            end
            
            
            train_data(row + counter_N*NTrain, :) = ynoisy_pam;
            train_label(row + counter_N*NTrain, 1) = count_classes;
       
        end
    end
    toc
end
disp("APSK is done")
save('./dataset/train_data_APSK.mat', 'train_data', '-mat');
save('./dataset/train_label_APSK.mat', 'train_label', '-mat');
disp("Files saved")

save('./dataset/train_data.mat', 'train_data', '-mat');
save('./dataset/train_label.mat', 'train_label', '-mat');
disp("Files saved")

%% Constellations Diagramms
constDiag_qam(ynoisy_qam)
constDiag_pam(ynoisy_pam)
constDiag_qpsk(ynoisy_qpsk)
constDiag_apsk(ynoisy_apsk)
        
