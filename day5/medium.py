# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 16:44
# @Author  : panfei
# @FileName: medium.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import numpy as np
import cv2

def medium_filter(img,k_size=3):
    W,H,C = img.shape

    #进行补零操作
    pad = k_size//2
    out = np.zeros((W+pad*2,H+pad*2,C),dtype=np.float)
    out[pad:pad+W,pad:pad+H]=img.copy().astype(np.float)

    tmp = out.copy()
    #用中值替换原来的值
    for y in range(W):
        for x in range(H):
            for c in range(C):
                out[pad+y,pad+x,c]=np.median(tmp[y:y+k_size,x:x+k_size,c])

    out = out.astype(np.uint8)

    return out

if __name__ == '__main__':
    img = cv2.imread('123.jpg')
    out = medium_filter(img)
    cv2.imshow('out',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
