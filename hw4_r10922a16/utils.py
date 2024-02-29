from cmath import pi, sqrt
from ctypes import sizeof
from random import randrange
from re import L
from statistics import mean
from turtle import shape, window_height
from unittest import result
from PIL import Image
from matplotlib import pyplot as plt 
from scipy import signal
from scipy import misc
import numpy as np
import cv2

def sampling(img):
    row, col = img.shape
    img_pad = np.pad(img,(1,1),'edge')
    size1, size2 = round(row/2), round(col/2)
    result = np.zeros((size1,size2))
    temp = 0
    idx1, idx2 = 0, 0
    for i in range(1,row,2):
        for j in range(1,col,2):
            array = [img_pad[i][j:j+2],img_pad[i+1][j:j+2]]
            temp = np.mean(array)
            result[idx1][idx2] = round(temp)
            temp = 0
            idx2 += 1
            if(idx2 >= size2):
                idx2 = 0
                idx1 += 1
    return result

def dithering(I,img):
    x, y = len(I), len(I[0])
    row, col = img.shape
    T =[]
    for i in range(x):
        for j in range(y):
            temp = 255.0*(I[i][j]+0.5)/x**2
            T.append(temp)
    T = np.reshape(T,(x,y))
    # print(T)
    i, j = 0, 0
    result = np.zeros((row,col))
    while(i < row and j < col):
        for idx1 in range(x):
            for idx2 in range(y):
                result[i+idx1][j+idx2] = img[i+idx1][j+idx2] > T[idx1][idx2]
        j += x
        if(j >= col):
            j = 0
            i += x

    for i in range(row):
        for j in range(col):
            if(result[i][j] == 1):
                result[i][j] = 255
            else:
                result[i][j] = 0
    return result

## only expandable from I2
def I_expand(I,power):
    x, y = len(I), len(I[0])
    I_new = I
    I_temp = I
    if(power <= 1):
        return I
    else:
        for i in range(power-1):
            ##block1
            I_new1 = [4*I_temp[idx1][idx2]+1 for idx1 in range(x) for idx2 in range(y)]
            I_new1 = np.reshape(I_new1,(x,y))
            ## block2
            I_new2 = [4*I_temp[idx1][idx2]+2 for idx1 in range(x) for idx2 in range(y)]
            I_new2 = np.reshape(I_new2,(x,y))
            ## block3
            I_new3 = [4*I_temp[idx1][idx2]+3 for idx1 in range(x) for idx2 in range(y)]
            I_new3 = np.reshape(I_new3,(x,y))
            ## block4
            I_new4 = [4*I_temp[idx1][idx2]   for idx1 in range(x) for idx2 in range(y)]
            I_new4 = np.reshape(I_new4,(x,y))

            I_new = np.block([[I_new1,I_new2],[I_new3,I_new4]])

            I_temp = I_new
            x *= 2
            y *= 2

        return I_new

def Floyd_Steinberg(img):
    row, col = img.shape
    threshold = 128
    img_pad = np.pad(img,(1,1),'edge')
    for i in range(1,col+1):
        for j in range(1,row):
            oldpixel = img_pad[i][j]
            newpixel = 0
            if(oldpixel > threshold):
                newpixel = 255
            img_pad[i][j] = newpixel
            error = oldpixel - newpixel
            
            temp = img_pad[i][j+1] + error *7 / 16
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i][j+1] = temp

            temp = img_pad[i+1][j-1] + error *3 / 16
            if(temp > 255):
                    temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+1][j-1] = temp

            temp = img_pad[i+1][j] + error *5 / 16
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+1][j] = temp

            temp = img_pad[i+1][j+1] + error *1 / 16
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+1][j+1] = temp
    result = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            result[i][j] = img_pad[i+1][j+1]
    return result

def Jarvis(img):
    row, col = img.shape
    threshold = 128
    img_pad = np.pad(img,(2,2),'constant')
    for i in range(2,col+1):
        for j in range(2,row+2):
            oldpixel = img_pad[i][j]
            newpixel = 0
            if(oldpixel > threshold):
                newpixel = 255
            img_pad[i][j] = newpixel
            error = oldpixel - newpixel

            temp = img_pad[i][j+1] + error *7 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i][j+1] = temp

            temp = img_pad[i][j+2] + error *5 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i][j+2] = temp

            temp = img_pad[i+1][j-2] + error *3 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+1][j-2] = temp
            
            temp = img_pad[i+1][j-1] + error *5 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+1][j-1] = temp

            temp = img_pad[i+1][j] + error *7 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+1][j] = temp

            temp = img_pad[i+1][j+1] + error *5 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+1][j+1] = temp

            temp = img_pad[i+1][j+2] + error *3 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+1][j+2] = temp

            temp = img_pad[i+2][j-2] + error *1 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+2][j-2] = temp

            temp = img_pad[i+2][j-1] + error *3 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+2][j-1] = temp

            temp = img_pad[i+2][j] + error *5 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+2][j] = temp

            temp = img_pad[i+2][j+1] + error *3 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+2][j+1] = temp

            temp = img_pad[i+2][j+2] + error *1 / 48
            if(temp > 255):
                temp = 255
            elif(temp < 0):
                temp =0
            img_pad[i+2][j+2] = temp

    result = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            result[i][j] = img_pad[i+2][j+2]
    return result

def gaussian_kernel(size, sigma):
    mu = np.floor([size / 2, size / 2])
    size = int(size)
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-(0.5/(sigma*sigma)) * (np.square(i-mu[0]) + 
            np.square(j-mu[0]))) / np.sqrt(2*np.pi*sigma*sigma)

    kernel = kernel/np.sum(kernel)
    return kernel

def unsharp_masking_in_freq_domain(img):
    row, col = img.shape
    c = 0.6
    kernel = gaussian_kernel(row, 3)
    kernel = np.abs(np.fft.fftshift(np.fft.fft2(kernel)))
    img_freq1 = np.fft.fftshift(np.fft.fft2(img))
    img_freq2 = img_freq1 * c / (2 * c - 1) - (1 - c) / (2 * c - 1) * img_freq1 * kernel 
    result = np.fft.ifft2(np.fft.ifftshift(img_freq2))

    result = result.real
    return result