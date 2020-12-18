# -*- coding: utf-8 -*-
# @Time    : 2020/10/12 14:04
# @Author  : panfei
# @FileName: backgroud.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2 as cv
import numpy as np

import cv2
import numpy as np
from matplotlib import pyplot as plt

import cv2 as cv
import numpy as np

img = cv2.imread('huahua3.png')
background = cv2.imread('huahua3.png')
h, w, ch = img.shape
OLD_IMG = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)
SIZE = (1, 65)
bgdModle = np.zeros(SIZE, np.float64)
fgdModle = np.zeros(SIZE, np.float64)
rect = (1, 1, img.shape[1], img.shape[0])
cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 10, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')



object = cv.bitwise_and(img, img, mask=mask2)

# 高斯模糊
se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
cv.dilate(mask2, se, mask2)
kernel = np.ones((15,15),np.uint8)
kernel_2 = np.ones((10,10),np.uint8)
mask2 = cv2.erode(mask2,kernel,iterations = 1)
mask2 = cv2.dilate(mask2,kernel_2,iterations = 1)
cv.imshow("result", mask2)
cv.waitKey(0)
cv.destroyAllWindows()


# 虚化背景
background = cv.GaussianBlur(background, (0, 0), 15)



result = np.zeros((h, w, ch), dtype=np.uint8)
for row in range(h):
    for col in range(w):
        w1 = mask2[row, col] / 255.0
        b, g, r = img[row, col]
        b1,g1,r1 = background[row, col]
        b = (1.0-w1) * b1 + b * w1
        g = (1.0-w1) * g1 + g * w1
        r = (1.0-w1) * r1 + r * w1
        result[row, col] = (b, g, r)

cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()
