import sys
import cv2
import unittest
import numpy as np
sys.path.append('..')
from common_func import sort_box_points

class Test(unittest.TestCase):
    def test_removal(self):
        TL = np.array([-2,2])
        BL = np.array([-2,-2])
        TR = np.array([2,2])
        BR = np.array([2,-2])
        box =np.array([BR,BL,TR,TL])
        print(sort_box_points(box))
        print("The correct order should be")
        print([BL,TL,TR,BR])
        
        #self.assertEqual(sort_box_points(box)[0].all(),[TL])
        
    
if __name__ == '__main__':
    unittest.main()