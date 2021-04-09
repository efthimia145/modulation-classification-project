
%% QAM 

M = 4; % Alphabet size, 16-QAM
x = randi([0 M-1],500,1); % Input signal

cpts = qammod(0:M-1,M);
constDiag_qam = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
    'XLimits',[-sqrt(M) sqrt(M)],'YLimits',[-sqrt(M) sqrt(M)]);

y = qammod(x,M);
ynoisy_qam = awgn(y,20,'measured'); % Noise addition (SNR)

z = qamdemod(ynoisy_qam,M);
[num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate

fprintf("Number of symbol errors: %f\n", num);
fprintf("Symbol error rate: %f\n", rt);

constDiag_qam(ynoisy_qam)

%% QPSK

M = 4; % Alphabet size, 16-QAM
x = randi([0 M-1],500,1); % Input signal

qpskmod = comm.QPSKModulator;
qpskdemod = comm.QPSKDemodulator;
cpts = qpskmod(x);

constDiag_qpsk = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
    'XLimits',[-sqrt(M) sqrt(M)],'YLimits',[-sqrt(M) sqrt(M)]);

y = qpskmod(x);
ynoisy_qpsk = awgn(y,20,'measured'); % Noise addition (SNR)

z = qpskdemod(ynoisy_qpsk);
[num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate

fprintf("Number of symbol errors: %f\n", num);
fprintf("Symbol error rate: %f\n", rt);

constDiag_qpsk(ynoisy_qpsk)

%% PAM

M = 16; % Alphabet size, 16-PAM
x = randi([0 M-1],500,1); % Input signal

cpts = pammod(x,M,pi/4);
constDiag_pam = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
    'XLimits',[-M/2 M/2],'YLimits',[-M/2 M/2]);

y = pammod(x,M,pi/4);
ynoisy_pam = awgn(y,20,'measured'); % Noise addition (SNR)

z = pamdemod(ynoisy_pam,M,pi/4);
[num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate


fprintf("Number of symbol errors: %f\n", num);
fprintf("Symbol error rate: %f\n", rt);
constDiag_pam(ynoisy_pam)

%% APSK 

M = [4 8 20]; % Alphabet size, APSK
radii = [0.3 0.7 1.2];
modOrder = sum(M);
x = randi([0 modOrder-1],500,1); % Input signal

cpts = apskmod(x,M,radii);
constDiag_apsk = comm.ConstellationDiagram('ReferenceConstellation',cpts, ...
    'XLimits',[-radii(3) radii(3)],'YLimits',[-radii(3) radii(3)]);

y = apskmod(x,M,radii);
ynoisy_apsk = awgn(y,20,'measured'); % Noise addition (SNR)

z = apskdemod(ynoisy_apsk,M,radii);
[num,rt] = symerr(x,z); % Compute number of symbol errors and symbol error rate


fprintf("Number of symbol errors: %f\n", num);
fprintf("Symbol error rate: %f\n", rt);
constDiag_apsk(ynoisy_apsk)