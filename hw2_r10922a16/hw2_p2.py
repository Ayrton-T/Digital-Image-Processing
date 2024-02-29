from cmath import pi, sin, sqrt
from re import L
from unittest import result
from PIL import Image
import numpy as np
import cv2


def sobel(img):

    threshold = 38
    op1 = [[-1,-2,-1],
            [0,0,0],
            [1,2,1]]
    op2 = [[-1,0,1],
            [-2,0,2],
            [-1,0,1]]
    row, col = img.shape
    result1 = np.copy(img)
    temp = np.pad(img,(2,2),'edge')

    for i in range(row):
        for j in range(col):
            temp1 = 0
            for k in range(3):
                for l in range(3):
                    temp1 += temp[i+1+k][j+1+l] * op1[k][l]
            temp2 = 0
            for k in range(3):
                for l in range(3):
                    temp2 += temp[i+1+k][j+1+l] * op2[k][l]
            result1[i][j] = np.round(sqrt(temp1*temp1 + temp2*temp2)/4)

    ans = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if(result1[i][j] >= threshold): ans[i][j] = 255
            else: ans[i][j] = 0

    return ans

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

def sample3_improve(img):
    row, col = img.shape
    temp = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if(img[i][j] != 0):
                temp[i][j] = 255
            else:
                temp[i][j] = 0

    # cv2.imwrite("result6_test.png",temp)
    temp2 = sobel(temp)
    # cv2.imwrite("result6_test2.png",temp2)
    for i in range(row):
        for j in range(col):
            if(temp[i][j] == temp2[i][j]):
                img[i][j] = 0
    # cv2.imwrite("result6_test3.png",img)

    img = transfer_function(img)
    # cv2.imwrite("result6_test4.png",img)

    return img

def cat_cat_friends(img):
    row, col = img.shape
    temp = np.zeros((row,col))
    ## scaling
    x = 0
    y = 0
    ## simply just use downsampling
    for i in range(0,row,2):
        for j in range(0,col,2):
            temp[x][y] = img[i][j]
            y += 1
        y = 1
        x += 1
    ## become 1/4 compare to original
    ## shift
    ## i think need to shift right 145 pixel down 20 pixel
    temp1 = np.zeros((row,col))
    for i in range(round(row/2)):
        for j in range(round(col/2)):
            temp1[i+20][j+145] = temp[i][j]
    # for i in range(round(row/2)):
    #     for j in range(round(col/2)):
    #         temp1[i+50][j] = temp[i][j] 
    ## rotate 90 degree
    temp2 = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            temp2[i][j] = temp1[599-j][i]
    for i in range(row):
        for j in range(col):
            if(temp2[i][j] != temp1[i][j] and temp1[i][j] != 0):
                temp2[i][j] = temp1[i][j]
    ## diagonal flip
    temp3 = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            temp3[i][j] = temp2[j][i]
    for i in range(row):
        for j in range(col):
            if(temp3[i][j] != temp2[i][j] and temp2[i][j] != 0):
                temp3[i][j] = temp2[i][j]
    return temp3

def liquid_cat(img):
    row, col = img1.shape
    A_c = 30
    A_r = 20
    omega = pi/75
    phi_c = 0.6*pi 
    phi_r = 0.8*pi
    result = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            q = i + A_c*sin(omega*j+phi_c)
            p = j + A_r*sin(omega*i+phi_r)
            q = round(q.real) + round(q.imag)
            p = round(p.real) + round(p.imag)
            if(p >= row): p = row-1
            if(q >= col): q = col-1
            if(p < 0): p = 0
            if(q < 0): q = 0
            result[q][p] = img[i][j]
    
    return result

## sample3
img1 = cv2.imread('imgs_2022\sample3.png', cv2.IMREAD_GRAYSCALE)
## sample5
img2 = cv2.imread('imgs_2022\sample5.png', cv2.IMREAD_GRAYSCALE)


result6 = sample3_improve(img1)
cv2.imwrite("result6.png",result6)
result7 = cat_cat_friends(img1)
cv2.imwrite("result7.png",result7)

result8 = liquid_cat(img2)
cv2.imwrite("result8.png",result8)
