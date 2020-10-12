# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 15:18
# @Author  : panfei
# @FileName: BGRTOGRAY.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2
import numpy as np

def BGR_TO_GRAY_WAY1(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b

    out = out.astype(np.uint8)

    print(out)


    return out

def BGR_TO_GRAY_WAY2(img):
    out = img.astype(np.uint8)
    out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
    return out


img = cv2.imread('imori.jpg').astype(np.float)
result = BGR_TO_GRAY_WAY1(img)
result1 = BGR_TO_GRAY_WAY2(img)
cv2.imshow('result',result)
cv2.imshow('result1',result1)
cv2.waitKey(0)
cv2.destroyAllWindows()