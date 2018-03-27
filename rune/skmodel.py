from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

class numberRecognizer:
    def __init__(self, model_dir="./model/finalized_model.sav"):
        self.mlp=pickle.load(open(model_dir, 'rb'))
        print("Model Loaded %s." % model_dir)
    def predict(self, predict_data):
        probs = self.mlp.predict_proba(predict_data)
        pred_number = self.mlp.predict(predict_data).astype(np.int64)
        print (probs.shape)
        print (pred_number.shape)
        return pred_number, probs
