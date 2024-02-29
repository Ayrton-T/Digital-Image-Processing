from PIL import Image
import numpy as np
import cv2

img = cv2.imread('hw1_sample_images\sample1.png')
row, col, channel  = img.shape
temp = np.copy(img)

for i in range(row):
    for j in range(col):
        img[i][j][0] = temp[i][col-j-1][0]
        img[i][j][1] = temp[i][col-j-1][1]
        img[i][j][2] = temp[i][col-j-1][2]


cv2.imwrite("result1.png",img)

color_img = np.asarray(Image.open('result1.png'))
img2 = np.mean(color_img, axis=2)


cv2.imwrite("result2.png",img2)