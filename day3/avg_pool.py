# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 16:03
# @Author  : panfei
# @FileName: avg_pool.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2
import numpy as np
def avg_pooling(img,G=8):
    out = img.copy()
    W,H,C= out.shape

#每8个像素点做一个平均池化
#计算宽高都有多少个块
    NW = int(W/G)
    NH = int(H/G)

#每个块的所有通道值都是该块通道的平均值
    for h in range(NH):
        for w in range(NW):
            for c in range(C):
                out[h*G:(h+1)*G,w*G:(w+1)*G,c]=np.mean(out[h*G:(h+1)*G,w*G:(w+1)*G,c].astype(np.int))

    return out

#最大池化是取该块通道的最大值，此处不再重复写


img = cv2.imread('imori.jpg')
out = avg_pooling(img)
cv2.imshow('out',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
