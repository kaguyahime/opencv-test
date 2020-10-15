# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 16:44
# @Author  : panfei
# @FileName: max_min_filter.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

#最大最小值滤波，当前值是 卷积核内最大值减去最小值

import cv2
import numpy as np

def max_min_filter(img,k_size=3):
    pad = k_size//2
    W,H=img.shape
    out = np.zeros((H+pad*2,W+pad*2),dtype=np.float)
    out[pad:pad+H,pad:pad+W]=img.copy().astype(np.float)

    tmp = out.copy()

    for y in range(H):
        for x in range(W):
            out[pad+y,pad+x]=np.max(tmp[y:y+k_size,x:x+k_size])-np.min(tmp[y:y+k_size,x:x+k_size])

    out = out[pad:pad+H,pad:pad+W].astype(np.uint8)
    return out




if __name__ == '__main__':
    img = cv2.imread('imori.jpg')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out = max_min_filter(gray_img)

    cv2.imshow('out',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()