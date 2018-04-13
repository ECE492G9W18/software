'''
Handwritten digits recognization model trainning and dumpping into file.
The finalized_model.sav is used for skmodel, when the sklearn is runnable.
The coef.sav is the output for MNIS_Model class, just using numpy to do the dirty job.
--Yiding Fan, 01/3/18 -- 27/3/18
'''
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
#The outout files
filename = 'finalized_model.sav'
coe='coef.sav'
def pre(mnist):
    retu=np.zeros(mnist.data.shape)
    print retu.shape
    for i in range(len(mnist.data)):
        ret,re=cv2.threshold(mnist.data[i], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        re.resize((1,784))
        retu[i]=re/ 255.
    return retu,mnist.target
if __name__ == "__main__":
#Loading the traing data from official website
    print("before loading")
    mnist = fetch_mldata("MNIST original")


# rescale the data, use the traditional train/test split
    X, y = pre(mnist)
    print("loaded")
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
#The training, we are using 4 layers neural network. That means the 28*28 images are in 1 d shape, which means
#they are (1,784) in numpy array. The first input layer is (784,28), the output will be (1,28) for each. Before
#entering the second layer, ReLu applied on the result. The Relu simply reduce the function Max(0,X) on each
#value in the array. The second layer is (28,28). The last one is (28,10). Therefore, we will have 10
#possibility for [0-9] digit. To predict the result, we take highest possibility as the result.
    mlp = MLPClassifier(hidden_layer_sizes=(28,28), max_iter=1400, alpha=1e-4,
                   solver='sgd', verbose=10,tol=1e-5, random_state=1,learning_rate_init=.01)
    mlp.fit(X_train, y_train)

#show the result of training
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))
#dump the result
    filename = 'finalized_model.sav'
    pickle.dump(mlp, open(filename, 'wb'))
    pickle.dump(mlp.coefs_,open(coe,'wb'))
    coef=pickle.load(open(coe, 'rb'))

