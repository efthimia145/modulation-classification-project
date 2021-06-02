import numpy as np
import matplotlib.pyplot as plt

class_tag = 0 # 0-bpsk; 1-4qam; 2-8psk; 3-16qam 

L = 500
load_path = 'pred_confusion_mat_L' + str(L) + '.npy'
data = np.load(load_path)

EsNoLow = 0
EsNoHigh = 50
Gap = 10
nClass = data.shape[1]

num_point = (int)((EsNoHigh-EsNoLow)/Gap) + 1
snr = np.zeros(num_point)
acc_4qam = np.zeros(num_point)
acc_8qam = np.zeros(num_point)
acc_16qam = np.zeros(num_point)
acc_64qam = np.zeros(num_point)
acc_4psk = np.zeros(num_point)
acc_2psk = np.zeros(num_point)
acc_4pam = np.zeros(num_point)
acc_8pam = np.zeros(num_point)
acc_16pam = np.zeros(num_point)
acc_64pam = np.zeros(num_point)
acc_4apsk = np.zeros(num_point)
acc_8apsk = np.zeros(num_point)
acc_16apsk = np.zeros(num_point)

cnt_sum = 0
for rows in data[0,:]:
    for ele in rows:
        cnt_sum = cnt_sum + ele
cnt_sum_single_class = cnt_sum/nClass

for i in range(num_point):
    snr[i] = EsNoLow + i*Gap
    acc_4qam[i] = data[i, class_tag, 0]/cnt_sum_single_class
    acc_8qam[i] = data[i, class_tag, 1]/cnt_sum_single_class
    acc_16qam[i] = data[i, class_tag, 2]/cnt_sum_single_class
    acc_64qam[i] = data[i, class_tag, 3]/cnt_sum_single_class
    acc_4psk[i] = data[i, class_tag, 4]/cnt_sum_single_class
    acc_2psk[i] = data[i, class_tag, 5]/cnt_sum_single_class
    acc_4pam[i] = data[i, class_tag, 6]/cnt_sum_single_class
    acc_8pam[i] = data[i, class_tag, 7]/cnt_sum_single_class
    acc_16pam[i] = data[i, class_tag, 8]/cnt_sum_single_class
    acc_64pam[i] = data[i, class_tag, 9]/cnt_sum_single_class
    acc_4apsk[i] = data[i, class_tag, 10]/cnt_sum_single_class
    acc_8apsk[i] = data[i, class_tag, 11]/cnt_sum_single_class
    acc_16apsk[i] = data[i, class_tag, 12]/cnt_sum_single_class

LineWidth = 1.0
for i in range(nClass):
    if i == 0:
        plt.plot(snr, acc_4qam, '*-', linewidth = LineWidth, label = '4QAM')
    elif i == 1:
        plt.plot(snr, acc_8qam, 'o-', linewidth = LineWidth, label = '8QAM')
    elif i == 2:
        plt.plot(snr, acc_16qam, '^-', linewidth = LineWidth, label = '16QAM')
    elif i == 3:
        plt.plot(snr, acc_64qam, '<-', linewidth = LineWidth, label = '64QAM')
    elif i == 4:
        plt.plot(snr, acc_4psk, '1-', linewidth = LineWidth, label = 'QPSK')
    elif i == 5:
        plt.plot(snr, acc_2psk, '2-', linewidth = LineWidth, label = 'BPSK')
    elif i == 6:
        plt.plot(snr, acc_4pam, '3-', linewidth = LineWidth, label = '4PAM')
    elif i == 7:
        plt.plot(snr, acc_8pam, '4-', linewidth = LineWidth, label = '8PAM')
    elif i == 8:
        plt.plot(snr, acc_16pam, 'x-', linewidth = LineWidth, label = '16PAM')
    elif i == 9:
        plt.plot(snr, acc_64pam, 'o-', linewidth = LineWidth, label = '64PAM')
    elif i == 10:
        plt.plot(snr, acc_4apsk, 'v-', linewidth = LineWidth, label = '4APSK')
    elif i == 11:
        plt.plot(snr, acc_8apsk, '<-', linewidth = LineWidth, label = '8APSK')
    elif i == 12:
        plt.plot(snr, acc_16apsk, '>-', linewidth = LineWidth, label = '16APSK')

plt.grid(True)
plt.legend(loc='lower right')
plt.xlim((EsNoLow, EsNoHigh))
plt.ylim((0,1))
plt.xlabel('SNR(dB)')
plt.ylabel('Pc')
#plt.title('CNN_AWGN_EveryClass(L = 100)')
plt.savefig('single_class_' + str(class_tag) + '_L' + str(L) + '.png', format='png')