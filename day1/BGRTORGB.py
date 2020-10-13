# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 14:35
# @Author  : panfei
# @FileName: BGRTORGB.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2

#opencv读取展示是bgr，matplotlib读取展示是RGB

def BGR_TO_RGB(img):
    b = img[:,:,0].copy()
    g = img[:,:,1].copy()
    r = img[:,:,2].copy()

    img[:,:,0] = r
    img[:,:,1] = g
    img[:,:,2] = b

    return img



img = cv2.imread('imori.jpg')
img = BGR_TO_RGB(img)
cv2.imwrite('out.jpg', img)
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()