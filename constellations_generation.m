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
maxDopplerShift  = 0; % Maximum Doppler shift of diffuse components (Hz)
delayVector1 = [0 110 190 410]*1e-9; % Discrete delays of four-path channel (s)
delayVector2 = delayVector1/2; % Discrete delays of four-path channel (s)
gainVector1  = [0 -9.7 -19.2 -22.8]; % Average path gains (dB)
gainVector2  = gainVector1/2; % Average path gains (dB)
pause_sec = 2;


% Configure a Rayleigh channel object
rayChan = comm.RayleighChannel(...
    'SampleRate',sampleRate500KHz, ...
    'PathDelays',delayVector2, ...
    'AveragePathGains',gainVector2, ...
    'NormalizePathGains',true, ...
    'MaximumDopplerShift',maxDopplerShift, ...
    'DopplerSpectrum',{doppler('Flat'),doppler('Flat'),doppler('Flat'),doppler('Flat')}, ...
    'RandomStream','mt19937ar with seed', ...
    'Seed',22, ...
    'PathGainsOutputPort',true);

    % doppler('Gaussian',0.6)

for SNR = 0:50:2
    for repeats = 0:50
        
        %% QAM 

        M = 4; % Alphabet size, 16-QAM
        bitsPerFrame_QAM = 500;

        [ynoisy_qam, constDiag_qam] = QAM(M, bitsPerFrame_QAM, rayChan, SNR);

        %% QPSK

        M = 4; % Alphabet size, 16-QAM
        bitsPerFrame_QPSK = 500;

        [ynoisy_qpsk, constDiag_qpsk] = QPSK(M, bitsPerFrame_QPSK, rayChan, SNR);

        %% PAM

        M = 16; % Alphabet size, 16-QAM
        bitsPerFrame_PAM = 500;

        [ynoisy_pam, constDiag_pam] = PAM(M, bitsPerFrame_PAM, rayChan, SNR);

        %% APSK 

        M = [4 8 20]; % Alphabet size, APSK
        radii = [0.3 0.7 1.2];
        bitsPerFrame_APSK = 500;

        [ynoisy_apsk, constDiag_apsk] = APSK(M, radii, bitsPerFrame_APSK, rayChan, SNR);

        %%
        constDiag_qam(ynoisy_qam)
        constDiag_pam(ynoisy_pam)
        constDiag_qpsk(ynoisy_qpsk)
        constDiag_apsk(ynoisy_apsk)

        %%
        p = scatterplot(ynoisy_qam);
        % set(gca,'Color','k')
        % grid on;
        saveas(gcf, append('QAM_', int2str(SNR), '_', int2str(repeats), '.png'))
        
        %%
        scatterplot(ynoisy_pam)
        % set(gca,'Color','k')
        % grid on;
        saveas(gcf, append('PAM_', int2str(SNR), '_', int2str(repeats), '.png'))
        
        %%
        scatterplot(ynoisy_qpsk)
        % set(gca,'Color','k')
        % grid on;
        saveas(gcf, append('QPSK_', int2str(SNR), '_', int2str(repeats), '.png'))
        
        %%
        scatterplot(ynoisy_apsk)
        % set(gca,'Color','k')
        % grid on;
        saveas(gcf, append('APSK_', int2str(SNR), '_', int2str(repeats), '.png'))
        
        pause(pause_sec);
    end
end
