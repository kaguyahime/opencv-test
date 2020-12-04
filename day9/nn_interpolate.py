# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 17:12
# @Author  : panfei
# @FileName: nn_interpolate.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2
import numpy as np

def interpolate(img,aw,ah):
    W, H, C = img.shape
    aW = int(W*aw)
    aH = int(H*ah)

    y = np.arange(aH).repeat(aW).reshape(aW,-1)
    x = np.tile(np.arange(aW),(aH, 1))
    y = np.round(y / aH).astype(np.int)
    x = np.round(x / aW).astype(np.int)

    out = img[y, x]

if __name__ == '__main__':

    img = cv2.imread('imori.jpg')
    interpolate(img,1,1)



