import numpy as np
import matplotlib.pyplot as plt

def plot_all_classes(data, num_point, SNR_low, SNR_high, SNR_step, nClass, plots_path):
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
        snr[i] = SNR_low + i*SNR_step
        acc_4qam[i] = data[i, 0, 0]/cnt_sum_single_class
        acc_8qam[i] = data[i, 1, 1]/cnt_sum_single_class
        acc_16qam[i] = data[i, 2, 2]/cnt_sum_single_class
        acc_64qam[i] = data[i, 3, 3]/cnt_sum_single_class
        acc_4psk[i] = data[i, 4, 4]/cnt_sum_single_class
        acc_2psk[i] = data[i, 5, 5]/cnt_sum_single_class
        acc_4pam[i] = data[i, 6, 6]/cnt_sum_single_class
        acc_8pam[i] = data[i, 7, 7]/cnt_sum_single_class
        acc_16pam[i] = data[i, 8, 8]/cnt_sum_single_class
        acc_64pam[i] = data[i, 9, 9]/cnt_sum_single_class
        acc_4apsk[i] = data[i, 10, 10]/cnt_sum_single_class
        acc_8apsk[i] = data[i, 11, 11]/cnt_sum_single_class
        acc_16apsk[i] = data[i, 12, 12]/cnt_sum_single_class

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
    plt.xlim((SNR_low, SNR_high))
    plt.ylim((0,1))
    plt.xlabel('SNR(dB)')
    plt.ylabel('Pc')
    plt.title('CNN_AWGN_Rayleigh (L = 500)')
    plt.savefig(plots_path + 'every_class_L' + str(L) + '.png', format='png')


def plot_accuracy(data, num_point, SNR_low, SNR_high, SNR_step, nClass, plots_path):
    snr = np.zeros(num_point)
    acc = np.zeros(num_point)
    cnt_sum = 0
    for rows in data[0,:]:
        for ele in rows:
            cnt_sum = cnt_sum + ele
    for i in range(num_point):
        snr[i] = SNR_low + i*SNR_step
        cnt_acc = 0
        for j in range(nClass):
            cnt_acc = cnt_acc + data[i, j, j]
        acc[i] = cnt_acc/cnt_sum

    plt.plot(snr, acc, 'o-', label = 'L = ' + str(L))

    plt.grid(True)
    #plt.legend(loc='lower right')
    plt.xlim((SNR_low, SNR_high))
    plt.ylim((0,1))
    plt.xlabel('SNR(dB)')
    plt.ylabel('Pc')
    plt.title('CNN_AWGN_Rayleigh')
    plt.savefig(plots_path + 'L' + str(L) + '.png', format='png')

def plot_single_class(data, num_point, SNR_low, SNR_step, SNR_high, nClass, plots_path):

    class_tag = 0 # 0-bpsk; 1-4qam; 2-8psk; 3-16qam

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
        snr[i] = SNR_low + i*SNR_step
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
    plt.xlim((SNR_low, SNR_high))
    plt.ylim((0,1))
    plt.xlabel('SNR(dB)')
    plt.ylabel('Pc')
    #plt.title('CNN_AWGN_EveryClass(L = 100)')
    plt.savefig(plots_path + 'single_class_' + str(class_tag) + '_L' + str(L) + '.png', format='png')


def plot_confusion_matrix(data, SNR_low, SNR_step, plots_path):
    
    SNR_main = 0

    cnt_sum = 0
    for rows in data[0,:]:
        for ele in rows:
            cnt_sum = cnt_sum + ele
    cnt_sum_single_class = cnt_sum/nClass

    idx = (SNR_main-SNR_low)//SNR_step
    predMat = data[idx,:]
    predMat = predMat/cnt_sum_single_class
    row = ['4QAM', '8QAM', '16QAM', '64QAM', 'QPSK', 'BPSK', '4PAM', '8PAM', '16PAM', '64PAM', '4-APSK', '8-APSK', '16-APSK']
    col = ['4QAM', '8QAM', '16QAM', '64QAM', 'QPSK', 'BPSK', '4PAM', '8PAM', '16PAM', '64PAM', '4-APSK', '8-APSK', '16-APSK']

    norm_conf = predMat

    fig = plt.figure(figsize=(18, 16), dpi=80)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(norm_conf, cmap=plt.cm.gray_r, 
                    interpolation='nearest')

    [width, height]= norm_conf.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(norm_conf[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), row)
    plt.yticks(range(height), col)
    savefilename = 'SNR=' + str(SNR_main) + 'dB L=' + str(L)
    plt.title(savefilename)
    plt.savefig(plots_path + savefilename + '.png', format='png')

L = 500
load_path = 'pred_confusion_mat_L' + str(L) + '.npy'
data = np.load(load_path)

SNR_low = 0
SNR_high = 50
SNR_step = 10
nClass = data.shape[1]
num_point = (int)((SNR_high-SNR_low)//SNR_step) + 1

plots_path = 'Plots/'

plot_all_classes(data, num_point, SNR_low, SNR_high, SNR_step, nClass, plots_path)
print("All classes plot -- printed")

plot_accuracy(data, num_point, SNR_low, SNR_high, SNR_step, nClass, plots_path)
print("Accuracy plot -- printed")

plot_single_class(data, num_point, SNR_low, SNR_step, SNR_high, nClass, plots_path)
print("Single class plot -- printed")

plot_confusion_matrix(data, SNR_low, SNR_step, plots_path)
print("Confusion Matrix plot -- printed")