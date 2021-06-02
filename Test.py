# coding=utf-8
from matplotlib import pyplot as plt
plt.style.use("ggplot")

import json
import numpy as np
import scipy.io as sio
import mat73

from keras import backend as K
from keras.models import model_from_json

# load model
model = model_from_json(json.load(open("model_struct.json")))

# load wights
model.load_weights("model_weights.h5")
              
# test begin
nClass = 13
seqLen = 500
EsNoLow = 0
EsNoHigh = 50
Gap = 10
pred_mat = np.zeros(((EsNoHigh-EsNoLow)//Gap + 1, nClass, nClass))
# load test data
data_mat = mat73.loadmat('test_data.mat')
data_complex = data_mat['test_data']
data_real = data_complex.real
data_imag = data_complex.imag
EsNoArray = data_real[:,-1]
y_test = data_real[:,-2]
data_real = data_real[:,0:seqLen]
data_imag = data_imag[:,0:seqLen]
data_real = data_real.reshape((data_real.shape[0], seqLen, 1))
data_imag = data_imag.reshape((data_imag.shape[0], seqLen, 1))
x_test = np.stack((data_real, data_imag), axis=1)
y_predict = model.predict(x_test)

# get predict result
print("*"*20)
print(y_test.shape[0])
print("*"*20)
for i in range(y_test.shape[0]):
    axis_0 = (int)((EsNoArray[i] - EsNoLow)/Gap)
    # should be
    axis_1 = (int)(y_test[i])
    # predict to be
    axis_2 = np.argmax(y_predict[i,:])
    pred_mat[axis_0, axis_1, axis_2] = pred_mat[axis_0, axis_1, axis_2] + 1
saveFileName  = "pred_confusion_mat_L" + str(seqLen)
np.save(saveFileName, pred_mat)