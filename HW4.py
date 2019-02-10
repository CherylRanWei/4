from __future__ import division
import cv2
import random
import function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte


print '************** Problem 1 **************'
'''
 Figure 4_1.bmp for implementation
'''

X = cv2.imread('4_1.bmp')
cv2.imwrite('4_1_original.jpg', X)
noise_img = function.sp_noise(X,0.15)
cv2.imwrite('4_1_noise.bmp', noise_img)
cv2.imwrite('4_1_noise.jpg', noise_img)

Y = cv2.imread('4_1_noise.bmp')

lpf_img = function.lpf(Y)
cv2.imwrite('4_1_lpf.jpg', lpf_img)

mpf_img = function.mpf(Y)
cv2.imwrite('4_1_mpf.jpg', mpf_img)


print '************** Problem 2 **************'
'''
 Figure 4_2.bmp for implementation
'''

X = cv2.imread('4_2.bmp')

hbf_img = function.hbf(X)
cv2.imwrite('4_2_mpf.jpg', hbf_img)












