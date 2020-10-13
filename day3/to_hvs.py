# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 13:52
# @Author  : panfei
# @FileName: to_hvs.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime


#HSV 即使用色相（Hue）、饱和度（Saturation）、明度（Value）来表示色彩的一种方式。


import cv2
import numpy as np
def BGR_TO_RGB(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    img[:,:,0]=r
    img[:,:,1]=g
    img[:,:,2]=b

    return img


img = cv2.imread('imori.jpg')
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
cv2.imshow('img_hsv',img_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

