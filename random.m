%% ======================================================================== %
%   Main function of constellations generation. 
%   This script will be used to generate 4 different modulations, using 
%   different SNR values, and different parameters for the Rayleigh
%   Channel. 
%   The final scatter plots will be used for the team to prepare the
%   dataset and, consequently, train a model for modulation classification
%   via DNN (or CNN).
%% ======================================================================== %

sampleRate1GHz = 1e6; % Sample rate of 1G Hz 
sampleRate500KHz = 500e3; % Sample rate of 500K Hz
sampleRate20KHz  = 20e3; % Sample rate of 20K Hz
maxDopplerShift  = 50; % Maximum Doppler shift of diffuse components (Hz)
maxDopplerShift_null = 0;
delayVector = [0 90 210 440]*1e-9; % Discrete delays of four-path channel (s)
gainVector  = [0 -10.7 -18.1 -24.4]; % Average path gains (dB)
pause_sec = 2;


% Configure a Rayleigh channel object
rayChan = comm.RayleighChannel(...
    'SampleRate',sampleRate500KHz, ...
    'PathDelays',delayVector, ...
    'AveragePathGains',gainVector, ...
    'NormalizePathGains',true, ...
    'MaximumDopplerShift',maxDopplerShift_null, ...
    'DopplerSpectrum',{doppler('Flat'),doppler('Flat'),doppler('Flat'),doppler('Flat')}, ...
    'RandomStream','mt19937ar with seed', ...
    'Seed',22, ...
    'PathGainsOutputPort',true);

rayChan_doppler = comm.RayleighChannel(...
    'SampleRate',sampleRate500KHz, ...
    'PathDelays',delayVector, ...
    'AveragePathGains',gainVector, ...
    'NormalizePathGains',true, ...
    'MaximumDopplerShift',maxDopplerShift, ...
    'DopplerSpectrum',{doppler('Flat'),doppler('Flat'),doppler('Flat'),doppler('Flat')}, ...
    'RandomStream','mt19937ar with seed', ...
    'Seed',22, ...
    'PathGainsOutputPort',true);

SNR = 20;
L = 500;
train_data = zeros(1, L);

%% QAM 

M = 4; % Alphabet size, 16-QAM
bitsPerFrame_QAM = L;

[ynoisy_qam, constDiag_qam] = QAM(M, bitsPerFrame_QAM, rayChan, SNR);

%% QPSK

M = 4; % Alphabet size, 16-QAM
bitsPerFrame_QPSK = L;

[ynoisy_qpsk, constDiag_qpsk] = QPSK(M, bitsPerFrame_QPSK, rayChan, SNR);

%% PAM

M = 16; % Alphabet size, 16-QAM
bitsPerFrame_PAM = L;

[ynoisy_pam, constDiag_pam] = PAM(M, bitsPerFrame_PAM, rayChan, SNR);

%% APSK 

M = [4 8 20]; % Alphabet size, APSK
radii = [0.3 0.7 1.2];
bitsPerFrame_APSK = L;

[ynoisy_apsk, constDiag_apsk] = APSK(M, radii, bitsPerFrame_APSK, rayChan, SNR);

%%

save('./dataset/train_data.mat', 'train_data', '-mat');

constDiag_qam(ynoisy_qam)
constDiag_pam(ynoisy_pam)
constDiag_qpsk(ynoisy_qpsk)
constDiag_apsk(ynoisy_apsk)

pause(pause_sec);