'''
    Handwritten digits recognizer, barely using the numpy to present the model.
    --Yiding Fan 01/03/18 - 27/03/18
    '''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np

coe='coef.sav'

class numberRecognizer:
    #initializer, note since we are not using the sklearn, the code should load the coef.sav correctly.
    def __init__(self, model_dir="./model/coef.sav"):
        self.coef_=pickle.load(open(model_dir, 'rb'))
        print("Coe Loaded %s." % model_dir)

    #predict function, which using the loaded coefs_ layer by layer to get the result.
    def predict(self, predict_data):
        #prepare the return value.pre_number is the result, and probs are the possibilities that may used in predicton selection.(we don't trust all the results)
        pred_number=np.array([])
        probs=np.array([])
        
        #the input data is in (n,784), so we have to deal with it one by one
        for i in predict_data:
            #layer by layer
            first=relu(np.dot(i,self.coef_[0]))
            second=relu(np.dot(first,self.coef_[1]))
            out = np.dot(second,self.coef_[2])
            #append the result into output
            probs=np.append(probs,out)
            pred_number=np.append(pred_number,np.array([np.argmax(out)]))
        #reshape the probs
        probs=probs.reshape((-1,10))
        pred_number=pred_number.astype(np.int64)
        return pred_number, probs

def relu(X):
    X[X<=0] = 0
    return X
