from PIL import Image
import numpy as np
from matplotlib import pyplot as plt 
import cv2
from PSNR import signal_to_noise

def medfilter3by3(img):
    row,col=img.shape
    img=np.pad(img,(1,1),'edge')
    filter=np.array([[1,1,1],
                     [1,1,1],
                     [1,1,1]])
    result=np.zeros((row,col),dtype=int)

    for i in range(1,row+1):
        for j in range(1,col+1):
            temp = []
            for x in range(3):
                for y in range(3):
                    temp.append(img[i-1+x][j-1+y]*filter[x][y])
            result[i-1][j-1] = np.median(temp)
    return result

def medfilter5by5(img):
    row,col=img.shape
    img=np.pad(img,(2,2),'edge')
    filter=np.array([[1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1]])
    result=np.zeros((row,col),dtype=int)

    for i in range(2,row+2):
        for j in range(2,col+2):
            temp = []
            for x in range(5):
                for y in range(5):
                    temp.append(img[i-2+x][j-2+y]*filter[x][y])
            result[i-2][j-2] = np.median(temp)
    return result


def boxfilter3by3(img):
    row,col=img.shape
    img=np.pad(img,(1,1),'edge')
    filter=np.array([[1,1,1],
                     [1,1,1],
                     [1,1,1]])
    coef=1.0/filter.sum()
    result=np.zeros((row,col),dtype=int)

    for i in range(1,row+1):
        for j in range(1,col+1):
            temp = 0.0
            for x in range(3):
                for y in range(3):
                    temp += img[i-1+x][j-1+y]*filter[x][y]
            result[i-1][j-1] = temp * coef
    return result

def boxfilter5by5(img):
    row,col=img.shape
    img=np.pad(img,(2,2),'edge')
    filter=np.array([[1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,1,1,1]])
    coef=1.0/filter.sum()
    result=np.zeros((row,col),dtype=int)

    for i in range(2,row+2):
        for j in range(2,col+2):
            temp = 0.0
            for x in range(5):
                for y in range(5):
                    temp += img[i-2+x][j-2+y]*filter[x][y]
            result[i-2][j-2] = temp * coef
    return result


def low_pass(img):
    b = 1
    filter = np.array([[1,b,1],
                    [b,b**2,b],
                    [1,b,1]])
    cof = 1.0/filter.sum()
    row,col = img.shape
    output = np.zeros((row,col))
    img = np.pad(img,(1,1),'edge')

    for i in range(1,row+1):
        for j in range(1,col+1):
            temp = 0
            for k in range(3):
                for l in range(3):
                    temp += cof * filter[k][l] * img[i-1+k][j-1+l]
            output[i-1][j-1] = temp

    return output

original = cv2.imread('hw1_sample_images\sample3.png', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('hw1_sample_images\sample4.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('hw1_sample_images\sample5.png', cv2.IMREAD_GRAYSCALE)
original
result1 = low_pass(img1)
result2 = medfilter3by3(img2)

cv2.imwrite("result10.png", result1)
cv2.imwrite("result11.png", result2)
psnr1 = signal_to_noise(result1, original)
psnr2 = signal_to_noise(result2, original)


print(psnr1)
print(psnr2)