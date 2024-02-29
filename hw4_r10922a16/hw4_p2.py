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

## down sampling to create alias
img2 = cv2.imread('hw4_sample_images\sample2.png', cv2.IMREAD_GRAYSCALE)
result5 = utils.sampling(img2)
cv2.imwrite("result5.png",result5)

img3 = cv2.imread('hw4_sample_images\sample3.png', cv2.IMREAD_GRAYSCALE)
result6 = utils.unsharp_masking_in_freq_domain(img3)
cv2.imwrite("result6.png",result6)