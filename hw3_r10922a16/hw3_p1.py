from cmath import pi, sqrt
from random import randrange
from unittest import result
from PIL import Image
import numpy as np
import cv2

def boundary_extract(img):
    mask = [[1,1,1],
            [1,1,1],
            [1,1,1]]
    row,col = img.shape
    img_temp = np.copy(img)
    result = np.zeros((row,col))
    img1 = np.copy(img)
    
    for i in range(row):
        for j in range(col):
            if(img[i][j] < 128):
                img_temp[i][j] = 0
            else:
                img_temp[i][j] = 1

    for i in range(2,row-1):
        for j in range(2,col-1):
            temp = 0
            for k in range(3):
                for l in range(3):
                    temp = temp + img_temp[i+k-1][j+l-1]
            if(temp == 9):
                result[i][j] = 255
            else:
                result[i][j] = 0

    for i in range(row):
        for j in range(col):
            img1[i][j] = img1[i][j] - result[i][j]
    
    cv2.imwrite("result1.png",img1)
    return


def hole_filling(img):

    row,col = img.shape
    img_temp = np.copy(img)
    img1 = np.zeros((row,col))
    img_complement = np.zeros((row+2,col+2))

    ## binarized
    for i in range(row):
        for j in range(col):
            if(img[i][j] < 128):
                img1[i][j] = 0
                img_complement[i+1][j+1] = 2
            else:
                img1[i][j] = 1
                img_complement[i+1][j+1] = 0

    connected = np.zeros((row+2,col+2))
    for i in range(row):
        for j in range(col):
            if(img[i][j] < 128):
                connected[i+1][j+1] = 1
            else:
                connected[i+1][j+1] = 0
    iter = 1
    for i in range(1,row):
        for j in range(1,col):
            if(connected[i][j] == 1):
                connected[i][j] = iter
                iter+=1
            else:
                connected[i][j] = 0
    min_temp = 0
    change = True
    iter = 0
    while(change):
        change = False
        for i in range(1,row):
            for j in range(1,col):
                temp_list = []
                if(connected[i][j] != 0):
                    temp_list.append(connected[i][j])
                    if(connected[i-1][j] != 0):
                        temp_list.append(connected[i-1][j])
                    if(connected[i][j-1] != 0):
                        temp_list.append(connected[i][j-1])
                    if(connected[i+1][j] != 0):
                        temp_list.append(connected[i+1][j])
                    if(connected[i][j+1] != 0):
                        temp_list.append(connected[i][j+1])
                    if(len(temp_list) != 0):
                        min_temp = min(temp_list)
                        if(min_temp < connected[i][j]):
                            change = True
                            connected[i][j] = min_temp
                temp_list.clear()
        for i in range(row,1,-1):
            for j in range(col,1,-1):
                temp_list = []
                if(connected[i][j] != 0):
                    temp_list.append(connected[i][j])
                    if(connected[i-1][j] != 0):
                        temp_list.append(connected[i-1][j])
                    if(connected[i][j-1] != 0):
                        temp_list.append(connected[i][j-1])
                    if(connected[i+1][j] != 0):
                        temp_list.append(connected[i+1][j])
                    if(connected[i][j+1] != 0):
                        temp_list.append(connected[i][j+1])
                    if(len(temp_list) != 0):
                        min_temp = min(temp_list)
                        if(min_temp < connected[i][j]):
                            change = True
                            connected[i][j] = min_temp
                temp_list.clear()

    for i in range(1,row):
        for j in range(1,col):
            if(connected[i][j] > 1):
                connected[i][j] = 255

    for i in range(row):
        for j in range(col):
            if(connected[i+1][j+1] == 255):
                img_temp[i][j] = 255

    for i in range(row):
        for j in range(col):
            if(img_complement[i+1][j+1] == 2):
                img_complement[i][j] = 255

    cv2.imwrite("result2.png",img_temp)
    # cv2.imwrite("complement.png",img_complement[1:row][1:col])
    
    return

def h(x0,x1,x2,x3):
    if(x0 == x1):
        if(x0 == x2 and x0 == x3):
            return 'r'
        else:
            return 'q'
    return 's'

def f(b,c,d,e):
    if(b == 'r' and c == 'r' and d == 'r' and e == 'r'):
        return 5
    cnt = 0
    if(b == 'q'): 
        cnt += 1
    if(c == 'q'):
        cnt += 1
    if(d == 'q'):
        cnt += 1
    if(e == 'q'):
        cnt += 1
    return cnt

def yokoi(ans):
    row,col = ans.shape
    table = np.zeros((row+2,col+2))
    for i in range(1,row+1):
        for j in range(1,col+1):
            table[i][j] = ans[i-1][j-1]

    for i in range(1,row):
        for j in range(1,col):
            if(table[i][j] == 1):
                b = h( table[i][j] , table[i][j+1] , table[i-1][j+1] , table[i-1][j] )
                c = h( table[i][j] , table[i-1][j] , table[i-1][j-1] , table[i][j-1] )
                d = h( table[i][j] , table[i][j-1] , table[i+1][j-1] , table[i+1][j] )
                e = h( table[i][j] , table[i+1][j] , table[i+1][j+1] , table[i][j+1] )
                ans[i-1][j-1] = f(b,c,d,e)
    return ans

def PRO(ans):
    row,col = ans.shape
    table = np.zeros((row+2,col+2))
    for i in range(1,row):
        for j in range(1,col):
            table[i][j] = ans[i-1][j-1]
    for i in range(1,row):
        for j in range(1,col):
            if(table[i][j] == 0):
                ans[i-1][j-1] = 0
            elif(table[i][j] == 1):
                if(table[i+1][j] == 1 or table[i-1][j] == 1 or table[i][j+1] == 1 or table[i][j-1] == 1):
                    ans[i-1][j-1] = 1
                else:
                    ans[i-1][j-1] = 2
            else:
                ans[i-1][j-1] = 2
    return ans

def CSO(ans):
    row,col = ans.shape
    table = np.zeros((row+2,col+2))
    for i in range(1,row):
        for j in range(1,col):
            table[i][j] = ans[i-1][j-1]
    for i in range(1,row):
        for j in range(1,col):
            if(table[i][j] != 0):
                table[i][j] = 1

    for i in range(row):
        for j in range(col):
            if(ans[i][j] == 1):
                b = h( table[i+1][j+1] , table[i+1][j+2] , table[i][j+2] , table[i][j+1] )
                c = h( table[i+1][j+1] , table[i][j+1] , table[i][j] , table[i+1][j] )
                d = h( table[i+1][j+1] , table[i+1][j] , table[i+2][j] , table[i+2][j+1] )
                e = h( table[i+1][j+1] , table[i+2][j+1] , table[i+2][j+2] , table[i+1][j+2] )
                cnt = 0
                if( b == 'q' ):
                    cnt+=1
                if( c == 'q' ):
                    cnt+=1
                if( d == 'q' ):
                    cnt+=1
                if( e == 'q' ):
                    cnt+=1

                if( cnt == 1 ):
                    ans[i][j] = 0
                    table[i+1][j+1] = 0	
            if(ans[i][j] != 0):
                ans[i][j] = 1
    return ans
    
def skeleton(img):
    row,col = img.shape
    ans = np.copy(img)
    for i in range(row):
        for j in range(col):
            if(ans[i][j] < 128):
                ans[i][j] = 0
            else:
                ans[i][j] = 1

    ans = yokoi(ans)
    ans = PRO(ans)
    ans = CSO(ans)

    change = True
    while(change):
        change = False
        temp = np.copy(ans)
        ans = yokoi(ans)
        ans = PRO(ans)
        ans = CSO(ans)
        for i in range(row):
            for j in range(col):
                if(temp[i][j] != ans[i][j]):
                    change = True
                    break
            if(change == True):
                break

    for i in range(row):
        for j in range(col):
            if(ans[i][j] == 1):
                ans[i][j] = 255
            else:
                ans[i][j] = 0

    return ans

def connected_component_labeling(img):
    row,col = img.shape
    connected = np.zeros((row+2,col+2))
    for i in range(row):
        for j in range(col):
            if(img[i][j] < 128):
                connected[i+1][j+1] = 0
            else:
                connected[i+1][j+1] = 1
    iter = 1
    for i in range(1,row):
        for j in range(1,col):
            if(connected[i][j] == 1):
                connected[i][j] = iter
                iter+=1
            else:
                connected[i][j] = 0
    min_temp = 0
    change = True
    iter = 0
    while(change):
        change = False
        for i in range(1,row):
            for j in range(1,col):
                temp_list = []
                if(connected[i][j] != 0):
                    temp_list.append(connected[i][j])
                    if(connected[i-1][j] != 0):
                        temp_list.append(connected[i-1][j])
                    if(connected[i][j-1] != 0):
                        temp_list.append(connected[i][j-1])
                    if(connected[i+1][j] != 0):
                        temp_list.append(connected[i+1][j])
                    if(connected[i][j+1] != 0):
                        temp_list.append(connected[i][j+1])
                    if(len(temp_list) != 0):
                        min_temp = min(temp_list)
                        if(min_temp < connected[i][j]):
                            change = True
                            connected[i][j] = min_temp
                temp_list.clear()
        for i in range(row,1,-1):
            for j in range(col,1,-1):
                temp_list = []
                if(connected[i][j] != 0):
                    temp_list.append(connected[i][j])
                    if(connected[i-1][j] != 0):
                        temp_list.append(connected[i-1][j])
                    if(connected[i][j-1] != 0):
                        temp_list.append(connected[i][j-1])
                    if(connected[i+1][j] != 0):
                        temp_list.append(connected[i+1][j])
                    if(connected[i][j+1] != 0):
                        temp_list.append(connected[i][j+1])
                    if(len(temp_list) != 0):
                        min_temp = min(temp_list)
                        if(min_temp < connected[i][j]):
                            change = True
                            connected[i][j] = min_temp
                temp_list.clear()
    
    label = 0
    object_num = []
    for i in range(1,row):
        for j in range(1,col):
            if(connected[i][j] > label):
                connected[i][j] = (connected[i][j] % 256)
                if(connected[i][j] < 30):
                    connected[i][j] += 45
                
    cv2.imwrite("temp.png",connected[1:row][1:col])
    temp = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)

    img_RGB = cv2.applyColorMap(temp, cv2.COLORMAP_INFERNO)
    cv2.imwrite("result5.png",img_RGB)

    return



## sample1
img = cv2.imread('hw3_sample_images\sample1.png', cv2.IMREAD_GRAYSCALE)
img_reverse = np.copy(img)
row ,col = img.shape
for i in range(row):
    for j in range(col):
        if(img_reverse[i][j] < 128):
            img_reverse[i][j] = 255
        else:
            img_reverse[i][j] = 0
hole_filling(img)
boundary_extract(img)
connected_component_labeling(img)
img3 = skeleton(img)
img4 = skeleton(img_reverse)
cv2.imwrite("result3.png",img3)
cv2.imwrite("result4.png",img4)