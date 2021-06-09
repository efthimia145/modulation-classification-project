# coding=utf-8
from matplotlib import pyplot as plt
plt.style.use("ggplot")

import json
import numpy as np
import scipy.io as sio
import mat73

from keras import backend as K
from keras.models import model_from_json


Nmodel = "model10"
# load model
model = model_from_json(json.load(open("Trained_weights/model_struct_" + Nmodel +".json")))

# load weights
model.load_weights("Trained_weights/model_weights_" + Nmodel +".h5")
              
# test begin
nClass = 13
L = 500
SNR_low = 0
SNR_high = 50
SNR_step = 10
pred_mat = np.zeros(((SNR_high-SNR_low)//SNR_step + 1, nClass, nClass))

# load test data
data_mat = mat73.loadmat('test_data.mat')
data_complex = data_mat['test_data']
data_real = data_complex.real
data_imag = data_complex.imag
EsNoArray = data_real[:,-1]
y_test = data_real[:,-2]
data_real = data_real[:,0:L]
data_imag = data_imag[:,0:L]
data_real = data_real.reshape((data_real.shape[0], L, 1))
data_imag = data_imag.reshape((data_imag.shape[0], L, 1))
x_test = np.stack((data_real, data_imag), axis=1)
y_predict = model.predict(x_test)

# get predict result
for i in range(y_test.shape[0]):
    axis_0 = (int)((EsNoArray[i] - SNR_low)/SNR_step)
    # should be
    axis_1 = (int)(y_test[i])
    # predict to be
    axis_2 = np.argmax(y_predict[i,:])
    pred_mat[axis_0, axis_1, axis_2] = pred_mat[axis_0, axis_1, axis_2] + 1

# save predict-confusion matrix
saveFileName  = "pred_confusion_mat_" + Nmodel +"_L" + str(L)
np.save(saveFileName, pred_mat)