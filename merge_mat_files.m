clc
clear

tic

NTrain = 5000;
NClasses = 12;
NSNRs = 6;
L = 500;
SNR_low = 0;
SNR_high = 50;
SNR_step = 10;

train_data = zeros(NTrain*NClasses*NSNRs, L);
train_label = zeros(NTrain*NClasses*NSNRs, 1);

length_train = NTrain*NClasses*NSNRs;
length_signal = NTrain;

counter_N = -1;

for count_classes = 0:11
    for SNR = SNR_low:SNR_step:SNR_high
        counter_N = counter_N + 1;
        
        disp(count_classes)
        disp(SNR)
        
        file = load(append('./dataset/train_data_', num2str(count_classes), '_', num2str(SNR),'.mat'));
        
        train_data(counter_N*NTrain+1:(counter_N+1)*NTrain, :) = file.signal_data;
        train_label(counter_N*NTrain+1:(counter_N+1)*NTrain, 1) = count_classes;
    end 
end

save('./dataset/train_data.mat', 'train_data', '-mat');
save('./dataset/train_label.mat', 'train_label', '-mat');

toc