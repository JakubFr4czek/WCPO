import cv2
import os
import requests
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

lenaSzum = cv2.imread('lenaRGBSzum.png')
lenaSzum = cv2.cvtColor(lenaSzum, cv2.COLOR_BGR2RGB)

lena = cv2.imread('lenaRGB.png')
lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)

def medianRGB(img, window_height = 8, window_width = 8):

    result_img = np.zeros(img.shape)

    for x in range(window_width, len(img) + 1, window_width):
        for y in range(window_height, len(img) + 1, window_height):
            window = img[x - window_width : x, y - window_height : y]
            
            d = np.zeros(window.shape)
            
            for w_x in range(len(window)):
                for w_y in range(len(window[w_x])):
                    d[w_x][w_y] = np.sqrt(np.sum(np.square(window - window[w_x, w_y])))
            
            idx_flat = np.argmin(d)
            idx = np.unravel_index(idx_flat, d.shape)
            
            for res_x in range(x - window_width, x):
                for res_y in range(y - window_width, y):
                    result_img[res_x][res_y] = img[x - window_width + idx[0], y - window_height + idx[1]]
    
    return result_img
        
filtered_img = medianRGB(lena.astype('int32')).astype('int16')

plt.imshow(filtered_img)
plt.show()