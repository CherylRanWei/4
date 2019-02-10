from __future__ import division
import cv2
import random
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.util.shape import view_as_windows
from skimage import img_as_ubyte


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - (prob/2)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            rdn = random.random()
            
            if rdn < (prob/2):
                output[i][j] = 0   # half prob for salt
            elif rdn > thres:
                output[i][j] = 255 # half prob for pepper
            else:
                output[i][j] = image[i][j]

    return output

def lpf(image):
    kernel = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                       [1/25, 1/25, 1/25, 1/25, 1/25],
                       [1/25, 1/25, 1/25, 1/25, 1/25],
                       [1/25, 1/25, 1/25, 1/25, 1/25],
                       [1/25, 1/25, 1/25, 1/25, 1/25]])
    
    output = np.copy(image[:,:,0])
    output = signal.convolve2d(output, kernel,mode='same', boundary='fill', fillvalue=0)

    return output

def mpf(image):
    rows = image.shape[0]
    cols = image.shape[1]

    X = np.copy(image[:,:,0])
    output = np.zeros((rows+2, cols+2))
    
    output[0:rows, 0:cols] = image[:,:,0]
    output[rows,:] = output[rows-1,:]
    output[rows+1,:] = output[rows-1,:]
    output[:,cols] = output[:,cols-1]
    output[:,cols+1] = output[:,cols-1]
 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            kernel = np.array([[output[i-2,j-2], output[i-2,j-1], output[i-2,j], output[i-2,j+1], output[i-2,j+2]],
                               [output[i-1,j-2], output[i-1,j-1], output[i-1,j], output[i-1,j+1], output[i-1,j+2]],
                               [output[i,  j-2], output[i,  j-1], output[i,  j], output[i,  j+1], output[i,  j+2]],
                               [output[i+1,j-2], output[i+1,j-1], output[i+1,j], output[i+1,j+1], output[i+1,j+2]],
                               [output[i+2,j-2], output[i+2,j-1], output[i+2,j], output[i+2,j+1], output[i+2,j+2]]])

            one = kernel.ravel()
            one.sort()
            output[i,j] = one[12]
            X = output[0:rows, 0:cols]
    
    return X


def conv(A, B):
    output = np.matmul(A, B)
    
    return output


def hbf(image):
    height = image.shape[0]   # number of rows
    width = image.shape[1]    # number of cols
    X = np.copy(image)

    window_shape = (3,3)
    kernel = np.array([-1,-1,-1,-1,9,-1,-1,-1,-1])
    #channel 0
    abstract_0 = view_as_windows(X[:,:,0], window_shape) #abstract every 3x3 (height-2, width-2, 3, 3)
    columns_0 = abstract_0.reshape(height-2, width-2, 9) #reshape to (height-2, width-2, 9)
    eq1_0 = columns_0.reshape((height-2)*(width-2), 9)   #reshape to ((height-2)*(width-2), 9) this is the matrix we need for equation (1)
    #channel 1
    abstract_1 = view_as_windows(X[:,:,1], window_shape) #abstract every 3x3 (height-2, width-2, 3, 3)
    columns_1 = abstract_1.reshape(height-2, width-2, 9) #reshape to (height-2, width-2, 9)
    eq1_1 = columns_1.reshape((height-2)*(width-2), 9)   #reshape to ((height-2)*(width-2), 9) this is the matrix we need for equation (1)
    #channel 2
    abstract_2 = view_as_windows(X[:,:,2], window_shape) #abstract every 3x3 (height-2, width-2, 3, 3)
    columns_2 = abstract_2.reshape(height-2, width-2, 9) #reshape to (height-2, width-2, 9)
    eq1_2 = columns_2.reshape((height-2)*(width-2), 9)   #reshape to ((height-2)*(width-2), 9) this is the matrix we need for equation (1)

    
    output_0 = conv(eq1_0, kernel) #convolution output for each channel
    output_1 = conv(eq1_1, kernel)
    output_2 = conv(eq1_2, kernel)

    
    final_0 = output_0.reshape(height-2,width-2) #reshape each channel
    final_0[final_0 < 0] = 0                     #replace negative by 0
    final_0[final_0 > 255] = 255                 #replace larger by 255
    
    final_1 = output_1.reshape(height-2,width-2) 
    final_1[final_1 < 0] = 0                     
    final_1[final_1 > 255] = 255                 

    final_2 = output_2.reshape(height-2,width-2) 
    final_2[final_2 < 0] = 0                    
    final_2[final_2 > 255] = 255

    X[0:height-2, 0:width-2, 0] = final_0
    X[0:height-2, 0:width-2, 1] = final_1
    X[0:height-2, 0:width-2, 2] = final_2

    return X





















    



