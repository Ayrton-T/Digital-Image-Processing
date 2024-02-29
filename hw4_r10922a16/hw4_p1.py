from cmath import pi, sqrt
from ctypes import sizeof
from random import randrange
from statistics import mean
from turtle import shape, window_height
from unittest import result
from PIL import Image
from scipy import signal
from scipy import misc
import numpy as np
import cv2
import utils

I2 = [[1.0, 2.0],
      [3.0, 0.0]]

##expand test
# print("I4:")
# I4 = utils.I_expand(I2,2)
# print(I4)
# print("I8:")
# I8 = utils.I_expand(I2,3)
# print(I8)

img1 = cv2.imread('hw4_sample_images\sample1.png', cv2.IMREAD_GRAYSCALE)
result1 = utils.dithering(I2,img1)
cv2.imwrite("result1.png",result1)

I256 = utils.I_expand(I2,8)
result2 = utils.dithering(I256,img1)
cv2.imwrite("result2.png",result2)

result3 = utils.Floyd_Steinberg(img1)
cv2.imwrite("result3.png",result3)

result4 = utils.Jarvis(img1)
cv2.imwrite("result4.png",result4)