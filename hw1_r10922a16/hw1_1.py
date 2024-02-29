from PIL import Image
import numpy as np
from matplotlib import pyplot as plt 
import cv2

def hist_eq(img):
    row,col = img.shape
    eq = np.copy(img)
    CDF = np.zeros(256)
    for i in range(row):
        for j in range(col):
            CDF[img[i][j]] += 1

    for i in range(1,256,1):
        CDF[i] += CDF[i-1]

    for i in range(256):
        if(CDF[i] != 0):
            min = CDF[i]
            break

    for i in range(row):
        for j in range(col):
            s_k = np.double(CDF[img[i][j]]-min) / np.double(row*col -min)
            eq[i][j] = np.round(s_k*255)

    return eq

def local_hist_eq(img):
    M, N = 10, 10
    mid = round((M*N)/2)
    t = 0
    for i in range(M):
        for j in range(N):
            t = t+1
            if(t == mid):
                Pad1 = i - 1
                Pad2 = j - 1
                break

    B = np.pad(img,(Pad1,Pad2),'constant')

    for i in range((np.size(B,0) - (Pad2*2)+1)):
        for j in range((np.size(B,1)-((Pad2*2)+1))):
            CDF = np.zeros(256)
            ele = 0
            inc = 1
            for x in range(M):
                for y in range(N):
                    if(inc == mid):
                        ele =B[i+x-1][j+y-1]+1
                    pos = B[i+x-1][j+x-1]+1
                    CDF[pos] = CDF[pos]+1
                    inc = inc + 1

            for l in range(2,256):
                CDF[l] = CDF[l]+CDF[l-1]
            img[i][j] = round(CDF[ele]/(M*N)*255)

    return img


def transfer_function(img):
    max_value = 220
    min_value = 10
    row, col = img.shape
    temp = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            if(img[i][j] >= max_value):
                temp[i][j] = 1
            elif(img[i][j] <= min_value):
                temp[i][j] = 0
            else:
                temp[i][j] = (img[i][j] - min_value)/(max_value - min_value)

    for i in range(row):
        for j in range(col):
            temp[i][j] = temp[i][j]*255
            

    return temp


img = cv2.imread('hw1_sample_images\sample2.png', cv2.IMREAD_GRAYSCALE)
img2 = np.copy(img)
row, col = img.shape
img1 = np.copy(img)

for i in range(row):
    for j in range(col):
        img1[i][j] = img[i][j] / 2

for i in range(row):
    for j in range(col):
        temp = img1[i][j]*3
        if(temp >= 255):
            img2[i][j] = 255
        else:
            img2[i][j] = temp

local1 = local_hist_eq(img1)
local2 = local_hist_eq(img2)
img5 = transfer_function(img)
eq1 = hist_eq(img1)
eq2 = hist_eq(img2)

cv2.imwrite("result3.png", img1)
cv2.imwrite("result4.png", img2)

cv2.imwrite("result7.png",local1)
cv2.imwrite("result8.png",local2)

cv2.imwrite('result9.png',img5)

cv2.imwrite("result5.png", eq1)
cv2.imwrite("result6.png", eq2)

## sample2
plt.hist(img, bins = range(255)) 
plt.title("histogram") 
plt.show()

## result3
plt.hist(img1, bins = range(255)) 
plt.title("histogram") 
plt.show()

## result4
plt.hist(img2, bins = range(255)) 
plt.title("histogram") 
plt.show()

## result5
plt.hist(eq1, bins = range(255)) 
plt.title("histogram") 
plt.show()

eq2 = cv2.imread('result6.png', cv2.IMREAD_GRAYSCALE)
local1 = cv2.imread('result7.png', cv2.IMREAD_GRAYSCALE)
local2 = cv2.imread('result8.png', cv2.IMREAD_GRAYSCALE)

## result6
plt.hist(eq2, bins = range(255)) 
plt.title("histogram") 
plt.show()

## result7
plt.hist(local1, bins = range(255)) 
plt.title("histogram") 
plt.show()

## result8
plt.hist(local2, bins = range(255)) 
plt.title("histogram") 
plt.show()

## result9
plt.hist(img5, bins = range(255)) 
plt.title("histogram") 
plt.show()
