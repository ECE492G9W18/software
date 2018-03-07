import sys
import cv2
import unittest
import numpy as np
sys.path.append('..')
from preprocess import cell_redundancy_removal

class Test(unittest.TestCase):
    def test_removal(self):
        A = ((489.4820556640625, 276.8758544921875), (34.9864387512207, 76.87226867675781), -60.68999481201172)
        B = ((488, 274), (33, 74), -55)
        rects= [A,B]
             
        self.assertEqual(cell_redundancy_removal(rects, rects, rects)[1],[A])
        print(cell_redundancy_removal(rects, rects, rects))
        
    
if __name__ == '__main__':
    unittest.main()