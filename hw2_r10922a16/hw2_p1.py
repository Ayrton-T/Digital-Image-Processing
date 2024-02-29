from cmath import pi, sqrt
from random import randrange
from unittest import result
from PIL import Image
import numpy as np
import cv2

def sobel(img):
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

    return result1


def sobel_canny(img):
    op1 = [[-1,-2,-1],
            [0,0,0],
            [1,2,1]]
    op2 = [[-1,0,1],
            [-2,0,2],
            [-1,0,1]]
    row, col = img.shape
    result1 = np.copy(img)
    Ix = np.zeros((row,col))
    Iy = np.zeros((row,col))
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
            Iy[i][j] = temp1
            Ix[i][j] = temp2
            result1[i][j] = np.round(sqrt(temp1*temp1 + temp2*temp2)/4)
    theta = np.arctan2(Iy, Ix)
    return result1,theta
    
def non_max_suppression(img, D):
    row, col = img.shape
    result = np.zeros((row,col))
    angle = D * 180. / pi
    angle[angle < 0] += 180

    for i in range(1,row-1):
        for j in range(1,col-1):
            q = 255
            r = 255
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i][j+1]
                r = img[i][j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1][j-1]
                r = img[i-1][j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1][j]
                r = img[i-1][j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1][j-1]
                r = img[i+1][j+1]
            if (img[i,j] >= q) and (img[i,j] >= r):
                result[i][j] = img[i][j]
            else:
                result[i][j] = 0

    return result

def double_thresholiding(img):
    high_threshold = 50
    low_threshold = 10
    row, col = img.shape
    result = np.zeros((row,col))
    ## 2 = edge pixel, 1 = candidate pixel, 0 = Non-edge pixel
    for i in range(row):
        for j in range(col):
            if(img[i][j] >= high_threshold):
                result[i][j] = 2
            elif(img[i][j] < high_threshold and img[i][j] >= low_threshold):
                result[i][j] = 1
            else:
                result[i][j] = 0
    return result

def connected_component_label(img):
    row, col = img.shape
    temp = np.pad(img,(1,1),'constant')

    for i in range(1, row+1):
        for j in range(1, col+1):
            if(temp[i][j] == 1):
                if(temp[i-1][j-1] == 2 or temp[i-1][j] == 2 or temp[i-1][j+1] == 2 
                or temp[i][j-1] == 2 or temp[i][j+1] or temp[i+1][j-1] == 2 or temp[i+1][j] == 2 or temp[i+1][j+1] == 2):
                    temp[i][j] = 2
                else:
                    temp[i][j] = 0
    result = temp[1:row+1 , 1:col+1]

    return result



def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return kernel


def canny(img):
    ## noise removal
    Gaussian_filter = gaussian_kernel(5,7)
    Gaussian_cof = 1 / Gaussian_filter.sum()
    row, col = img.shape
    result = np.zeros((row,col))
    step1 = np.copy(img)
    step1_pad = np.pad(img,(2,2),'edge')
    
    for i in range(2,row+2):
        for j in range(2,col+2):
            temp = 0
            for k in range(5):
                for l in range(5):
                    temp += Gaussian_filter[k][l] * step1_pad[i+k-2][j+l-2] * Gaussian_cof
            step1[i-2][j-2] = temp
    cv2.imwrite("canny_step1.png",step1)
    ## use sobel to calculate gradient
    step2, theta = sobel_canny(step1)
    cv2.imwrite("canny_step2.png",step2)
    ## non-maximal suppression
    step3 = non_max_suppression(step2,theta)
    cv2.imwrite("canny_step3.png",step3)
    ## Hysteretic thresholding(double thresholding)
    step4 = double_thresholiding(step3)
    ## connected component labeling method
    step5 = connected_component_label(step4)

    for i in range(row):
        for j in range(col):
            if(step5[i][j] == 2):
                result[i][j] = 255
            else:
                result[i][j] = 0

    return result

def Laplacian_of_Gaussian(img):    
    mask = [[ 0,  0,  0, -1, -1, -2, -1, -1,  0,  0,  0],
            [ 0,  0, -2, -4, -8, -9, -8, -4, -2,  0,  0],
            [ 0, -2, -7,-15,-22,-23,-22,-15, -7, -2,  0],
            [-1, -4,-15,-24,-14, -1,-14,-24,-15, -4, -1],
            [-1, -8,-22,-14, 52,103, 52,-14,-22, -8, -1],
            [-2, -9,-23, -1,103,178,103, -1,-23, -9, -2],
            [-1, -8,-22,-14, 52,103, 52,-14,-22, -8, -1],
            [-1, -4,-15,-24,-14, -1,-14,-24,-15, -4, -1],
            [ 0, -2, -7,-15,-22,-23,-22,-15, -7, -2,  0],
            [ 0,  0, -2, -4, -8, -9, -8, -4, -2,  0,  0],
            [ 0,  0,  0, -1, -1, -2, -1, -1,  0,  0,  0]]
    threshold = 3000
    row, col = img.shape
    img_pad = np.pad(img,(6,6),'edge')
    temp = np.zeros((row,col))
    result = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            temp0 = 0
            for a in range(-5,6):
                for b in range(-5,6):
                    temp0 += img_pad[i+5+a][j+5+b] * mask[a+5][b+5]
            if(temp0 >= threshold): temp[i][j] = 1
            elif(temp0 <= -threshold): temp[i][j] = -1
            else: temp[i][j] = 0

    img_pad2 = np.pad(temp,(1,1),'edge')
    for i in range(row):
        for j in range(col):
            flag = False
            if(img_pad2[i+1][j+1] == 1):
                for a in range(-1,1):
                    for b in range(-1,1):
                        if(img_pad2[i+1+a][j+1+b] == -1):
                            result[i][j] = 0
                            flag = True
                            break
                    if(flag == True): break
                    if(flag == False): result[i][j] = 255
            else: result[i][j] = 255

    return result


def edge_crispening(img):
    b = 1
    low_pass_filter = [[1,b,1],
                        [b,b**2,b],
                        [1,b,1]]
    low_pass_sum = np.sum(low_pass_filter)
    c = 5/6
    all_pass_factor = c / (2*c - 1)
    low_pass_factor = (1-c) / (2*c -1)
    row, col = img.shape

    temp1 = np.zeros((row,col))
    temp2 = np.pad(img,(2,2),'edge')

    temp3 = np.zeros((row,col))
    result = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            temp = 0
            for k in range(-1,1):
                for l in range(-1,1):
                    temp1[i][j] += temp2[i+1+k][j+1+l] * low_pass_filter[k+1][l+1] / low_pass_sum
            temp1[i][j] = temp1[i][j] * low_pass_factor
            temp3[i][j] = img[i][j] * all_pass_factor
            result[i][j] = temp3[i][j] - temp1[i][j]

    return result

def edge_crispening_test(img):
    b = 1
    high_pass_filter = [[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]]
    row, col = img.shape

    temp2 = np.pad(img,(2,2),'edge')

    result = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            for k in range(-1,1):
                for l in range(-1,1):
                    result[i][j] += temp2[i+1+k][j+1+l] * high_pass_filter[k+1][l+1]
    return result

## sample1
img1 = cv2.imread('imgs_2022\sample1.png', cv2.IMREAD_GRAYSCALE)
## sample2
img2 = cv2.imread('imgs_2022\sample2.png', cv2.IMREAD_GRAYSCALE)

row, col = img1.shape
temp = np.copy(img1)
result2 = np.copy(img1)
threshold = 38

result1 = sobel(img1)
## gradient image
cv2.imwrite("result1.png",result1)

for i in range(row):
    for j in range(col):
        if(result1[i][j] >= threshold): result2[i][j] = 255
        else: result2[i][j] = 0
## edge map
cv2.imwrite("result2.png",result2)

result3 = canny(img1)
cv2.imwrite("result3.png",result3)

result4 = Laplacian_of_Gaussian(img1)
cv2.imwrite("result4.png",result4)

result5 = edge_crispening(img2)
# test = edge_crispening_test(img2)
cv2.imwrite("result5.png",result5)
# cv2.imwrite("test_crispening.png",test)
