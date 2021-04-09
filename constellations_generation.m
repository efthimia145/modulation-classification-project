sampleRate1GHz = 1e6; % Sample rate of 1G Hz 
sampleRate500KHz = 500e3; % Sample rate of 500K Hz
sampleRate20KHz  = 20e3; % Sample rate of 20K Hz
maxDopplerShift  = 0; % Maximum Doppler shift of diffuse components (Hz)
delayVector = [0 110 190 410]*1e-9; % Discrete delays of four-path channel (s)
gainVector  = [0 -9.7 -19.2 -22.8]; % Average path gains (dB)
SNR = 20;

% Configure a Rayleigh channel object
rayChan = comm.RayleighChannel(...
    'SampleRate',sampleRate500KHz, ...
    'PathDelays',delayVector, ...
    'AveragePathGains',gainVector, ...
    'NormalizePathGains',true, ...
    'MaximumDopplerShift',maxDopplerShift, ...
    'DopplerSpectrum',{doppler('Flat'),doppler('Flat'),doppler('Flat'),doppler('Flat')}, ...
    'RandomStream','mt19937ar with seed', ...
    'Seed',22, ...
    'PathGainsOutputPort',true);

    % doppler('Gaussian',0.6)
    
%% QAM 

M = 4; % Alphabet size, 16-QAM
bitsPerFrame_QAM = 500;
x = randi([0 M-1],bitsPerFrame_QAM,1); % Input signal

cpts = qammod(0:M-1,M);
constDiag_qam = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
    'XLimits',[-M M],'YLimits',[-M M]);

y = qammod(x,M);
ynoisy_qam = awgn(y,SNR,'measured'); % Noise addition (SNR)

% release(rayChan)
ynoisy_qam = rayChan(ynoisy_qam);

z = qamdemod(ynoisy_qam,M);
[num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate

fprintf("Number of symbol errors: %f\n", num);
fprintf("Symbol error rate: %f\n", rt);

constDiag_qam(ynoisy_qam)

%% QPSK

M = 4; % Alphabet size, 16-QAM
bitsPerFrame_QPSK = 500;
x = randi([0 M-1],bitsPerFrame_QPSK,1); % Input signal

qpskmod = comm.QPSKModulator;
qpskdemod = comm.QPSKDemodulator;
cpts = qpskmod(x);

constDiag_qpsk = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
    'XLimits',[-M M],'YLimits',[-M M]);

y = qpskmod(x);
ynoisy_qpsk = awgn(y,SNR,'measured'); % Noise addition (SNR)

% release(rayChan)
ynoisy_qpsk = rayChan(ynoisy_qpsk);

z = qpskdemod(ynoisy_qpsk);
[num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate

fprintf("Number of symbol errors: %f\n", num);
fprintf("Symbol error rate: %f\n", rt);

constDiag_qpsk(ynoisy_qpsk)

%% PAM

M = 16; % Alphabet size, 16-QAM
bitsPerFrame_PAM = 500;
x = randi([0 M-1],bitsPerFrame_PAM,1); % Input signal

cpts = pammod(x,M,pi/4);
constDiag_pam = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
    'XLimits',[-M M],'YLimits',[-M M]);

y = pammod(x,M,pi/4);
ynoisy_pam = awgn(y,SNR,'measured'); % Noise addition (SNR)

% release(rayChan)
ynoisy_pam = rayChan(ynoisy_pam);

z = pamdemod(ynoisy_pam,M,pi/4);
[num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate


fprintf("Number of symbol errors: %f\n", num);
fprintf("Symbol error rate: %f\n", rt);
constDiag_pam(ynoisy_pam)

%% APSK 

M = [4 8 20]; % Alphabet size, APSK
radii = [0.3 0.7 1.2];
modOrder = sum(M);
bitsPerFrame_APSK = 500;
x = randi([0 modOrder-1],bitsPerFrame_APSK,1); % Input signal

cpts = apskmod(x,M,radii);
constDiag_apsk = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
    'XLimits',[-2*radii(3) 2*radii(3)],'YLimits',[-2*radii(3) 2*radii(3)]);

y = apskmod(x,M,radii);
ynoisy_apsk = awgn(y,SNR,'measured'); % Noise addition (SNR)

% release(rayChan)
ynoisy_apsk = rayChan(ynoisy_apsk);

z = apskdemod(ynoisy_apsk,M,radii);
[num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate


fprintf("Number of symbol errors: %f\n", num);
fprintf("Symbol error rate: %f\n", rt);
constDiag_apsk(ynoisy_apsk)