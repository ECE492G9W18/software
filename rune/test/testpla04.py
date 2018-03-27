import os
import sys
import cv2
import unittest
import numpy as np
sys.path.append('..')
from mnist_model import *

class Test(unittest.TestCase):
    def test_recog(self):
        img = cv2.imread("7_/_00038.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
        test_data = (255 - img.reshape((1, img.shape[0] * img.shape[1])) )/ 255.
        
        recognizer = numberRecognizer(model_dir="../model/coef.sav")
        self.assertEqual(recognizer.predict(test_data)[0][0],7)
        print(recognizer.predict(test_data))

if __name__ == '__main__':
    unittest.main()

