import numpy as np
import matplotlib.pyplot as plt

EsNo = 50

L = 500
load_path = 'pred_confusion_mat_L' + str(L) + '.npy'
data = np.load(load_path)

EsNoLow = 0
EsNoHigh = 50
Gap = 10
nClass = data.shape[1]
cnt_sum = 0
for rows in data[0,:]:
    for ele in rows:
        cnt_sum = cnt_sum + ele
cnt_sum_single_class = cnt_sum/nClass

idx = (int)((EsNo-EsNoLow)/Gap)
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
savefilename = 'EsNo=' + str(EsNo) + 'dB L=' + str(L)
plt.title(savefilename)
plt.savefig(savefilename + '.png', format='png')