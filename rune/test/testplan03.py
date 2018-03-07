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
        box =np.array([TL,BL,TR,BR])
        #print(sort_box_points(box))
        
        self.assertEqual(sort_box_points(box)[0].all(),[TL])
        
    
if __name__ == '__main__':
    unittest.main()