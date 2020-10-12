# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 16:28
# @Author  : panfei
# @FileName: BGRTOBIN.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2
import numpy as np

def BGR_TO_BIN(img):
    img = img.astype(np.uint8)
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1[img1 < 128] = 0
    img1[img1 >= 128] = 255

    return img1

img = cv2.imread('imori.jpg')
out = BGR_TO_BIN(img)
cv2.imshow('out',out)
cv2.waitKey(0)
cv2.destroyAllWindows()