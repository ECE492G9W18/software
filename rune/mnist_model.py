from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np
coe='coef.sav'
class numberRecognizer:
    def __init__(self, model_dir="./model/coef.sav"):
        self.coef_=pickle.load(open(model_dir, 'rb'))
        print("Coe Loaded %s." % model_dir)
    def predict(self, predict_data):
        pred_number=np.array([])
        probs=np.array([])
        for i in predict_data:
            first=relu(np.dot(i,self.coef_[0]))
            second=relu(np.dot(first,self.coef_[1]))
            out = np.dot(second,self.coef_[2])
            probs=np.append(probs,out)
            pred_number=np.append(pred_number,np.array([np.argmax(out)]))
        probs=probs.reshape((-1,10))
        pred_number=pred_number.astype(np.int64)
        return pred_number, probs
def relu(X):
    X[X<=0] = 0
    return X
