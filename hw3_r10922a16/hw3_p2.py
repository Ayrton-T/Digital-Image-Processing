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

def Laws(img):
    Laws1 = [[1.0/36,2.0/36,1.0/36],
            [ 2.0/36,4.0/36,2.0/36],
            [ 1.0/36,2.0/36,1.0/36]]
    Laws2 = [[1.0/12,2.0/12,1.0/12],
            [ 2.0/12,4.0/12,2.0/12],
            [ 1.0/12,2.0/12,1.0/12]]
    Laws3 = [[1.0/12,0.0/12,-1.0/12],
            [ 2.0/12,0.0/12,-2.0/12],
            [ 1.0/12,0.0/12,-1.0/12]]
    Laws4 = [[-1.0/12,2.0/12,-1.0/12],
            [ -2.0/12,4.0/12,-2.0/12],
            [ -1.0/12,2.0/12,-1.0/12]]
    Laws5 = [[ 1.0/4,0.0/4,-1.0/4],
            [  0.0/4,0.0/4, 0.0/4],
            [ -1.0/4,0.0/4, 1.0/4]]
    Laws6 = [[1.0/4, 2.0/4,-1.0/4],
            [ 0.0/4, 0.0/4, 0.0/4],
            [ 1.0/4,-2.0/4, 1.0/4]]
    Laws7 = [[-1.0/12,-2.0/12,-1.0/12],
            [  2.0/12, 4.0/12, 2.0/12],
            [ -1.0/12,-2.0/12,-1.0/12]]
    Laws8 = [[-1.0/4, 0.0/4, 1.0/4],
            [  2.0/4, 0.0/4,-2.0/4],
            [ -1.0/4, 0.0/4, 1.0/4]]
    Laws9 = [[ 1.0/4,-2.0/4, 1.0/4],
            [ -2.0/4, 4.0/4,-2.0/4],
            [  1.0/4,-2.0/4, 1.0/4]]
    window_width = 13
    window_height = 13
    
    row,col = img.shape
    img_copy = np.copy(img)
    M1 = np.zeros((row,col))
    M2 = np.zeros((row,col))
    M3 = np.zeros((row,col))
    M4 = np.zeros((row,col))
    M5 = np.zeros((row,col))
    M6 = np.zeros((row,col))
    M7 = np.zeros((row,col))
    M8 = np.zeros((row,col))
    M9 = np.zeros((row,col))

    M1 = signal.convolve2d(img_copy, Laws1, boundary='symm', mode='same')
    M2 = signal.convolve2d(img_copy, Laws2, boundary='symm', mode='same')
    M3 = signal.convolve2d(img_copy, Laws3, boundary='symm', mode='same')
    M4 = signal.convolve2d(img_copy, Laws4, boundary='symm', mode='same')
    M5 = signal.convolve2d(img_copy, Laws5, boundary='symm', mode='same')
    M6 = signal.convolve2d(img_copy, Laws6, boundary='symm', mode='same')
    M7 = signal.convolve2d(img_copy, Laws7, boundary='symm', mode='same')
    M8 = signal.convolve2d(img_copy, Laws8, boundary='symm', mode='same')
    M9 = signal.convolve2d(img_copy, Laws9, boundary='symm', mode='same')

    pad_width = 6
    pad_height = 6
    energy1 = np.pad(M1,(pad_width,pad_height),'symmetric')
    energy2 = np.pad(M2,(pad_width,pad_height),'symmetric')
    energy3 = np.pad(M3,(pad_width,pad_height),'symmetric')
    energy4 = np.pad(M4,(pad_width,pad_height),'symmetric')
    energy5 = np.pad(M5,(pad_width,pad_height),'symmetric')
    energy6 = np.pad(M6,(pad_width,pad_height),'symmetric')
    energy7 = np.pad(M7,(pad_width,pad_height),'symmetric')
    energy8 = np.pad(M8,(pad_width,pad_height),'symmetric')
    energy9 = np.pad(M9,(pad_width,pad_height),'symmetric')

    T1 = np.zeros((row,col))
    T2 = np.zeros((row,col))
    T3 = np.zeros((row,col))
    T4 = np.zeros((row,col))
    T5 = np.zeros((row,col))
    T6 = np.zeros((row,col))
    T7 = np.zeros((row,col))
    T8 = np.zeros((row,col))
    T9 = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            for k in range(window_width):
                for l in range(window_height):
                    T1[i][j] += energy1[i+k][j+l]**2
                    T2[i][j] += energy2[i+k][j+l]**2
                    T3[i][j] += energy3[i+k][j+l]**2
                    T4[i][j] += energy4[i+k][j+l]**2
                    T5[i][j] += energy5[i+k][j+l]**2
                    T6[i][j] += energy6[i+k][j+l]**2
                    T7[i][j] += energy7[i+k][j+l]**2
                    T8[i][j] += energy8[i+k][j+l]**2
                    T9[i][j] += energy9[i+k][j+l]**2

    FM = np.zeros((9, row*col))
    for j in range(row*col):
        cur_i = np.int(np.mod(j - 1, row))
        cur_j = np.int(np.ceil(j / row)-1)
        FM[0][j] = T1[cur_i][cur_j]
        FM[1][j] = T2[cur_i][cur_j]
        FM[2][j] = T3[cur_i][cur_j]
        FM[3][j] = T4[cur_i][cur_j]
        FM[4][j] = T5[cur_i][cur_j]
        FM[5][j] = T6[cur_i][cur_j]
        FM[6][j] = T7[cur_i][cur_j]
        FM[7][j] = T8[cur_i][cur_j]
        FM[8][j] = T9[cur_i][cur_j]

    FP = np.zeros((3*row,3*col))
    for i in range(row):
        for j in range(col):
            FP[i][j] = T1[i][j]
            FP[i][j+col] = T2[i][j]
            FP[i][j+2*col] = T3[i][j]
            FP[i+row][j] = T4[i][j]
            FP[i+row][j+col] = T5[i][j]
            FP[i+row][j+2*col] = T6[i][j]
            FP[i+2*row][j] = T7[i][j]
            FP[i+2*row][j+col] = T8[i][j]
            FP[i+2*row][j+2*col] = T9[i][j]
    cv2.imwrite("FP.png",FP)

    return FM

# def classify(FM, k, m ,n):
#     idx = k_means(np.transpose(FM), k)

#     I = np.zeros((m, n))
#     for i in range(m*n):
#         cur_i = np.int(np.mod(i - 1, m))
#         cur_j = np.int(np.ceil(i / m)-1)
#         if (idx[i] == 1):
#             I[cur_i][cur_j] = 0
#         elif(idx[i] == 2):
#             I[cur_i][cur_j] = 80
#         elif(idx[i] == 3):
#             I[cur_i][cur_j] = 160
#         else:
#             I[cur_i][cur_j] = 240
#     return I

# def k_means(data,k):
#     nData, unuse = data.shape
#     centers = data[np.random.randint(1,nData, k),:]
#     idx = np.zeros((nData, 1))
#     idx_prev = np.ones((nData, 1))

#     isDiff = sum(sum(idx != idx_prev)) > 0
#     while(isDiff):
#         idx_prev = idx
#         for i in range(nData):
#             distSquare = 0
#             for j in range(len(data[i])):
#                 distSquare += sum((centers-data[i][j])**2)
#             idx[i] = min(distSquare)

#         for i in range(k):
#             index = np.argwhere(idx == i)
#             centers[i,:] = np.mean(data[index,:])
#         isDiff = sum(sum(idx != idx_prev)) > 0

#     return idx

# def classification_improve(img):
#     pass


## sample2
img = cv2.imread('hw3_sample_images\sample2.png', cv2.IMREAD_GRAYSCALE)
feature = Laws(img)
# row,col = img.shape
# k = 4
# feature = cv2.imread('FM.png', cv2.IMREAD_GRAYSCALE)

# L = classify(feature, k, 600, 900)



