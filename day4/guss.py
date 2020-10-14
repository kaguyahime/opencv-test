# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 16:03
# @Author  : panfei
# @FileName: avg_pool.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime


#高斯滤波器将中心像素周围的像素按照高斯分布加权平均进行平滑化。这样的（二维）权值通常被称为卷积核或者滤波器。
import cv2
import numpy as np

def guss_filter(img,k_size=3,sigma=1.3):
    if len(img.shape)==3:
        W,H,C=img.shape
    else:
        img = np.expand_dims(img,axis=-1)# axis 0,1,2 从高维开始扩展，-1代表扩展最后一个
        W,H,C=img.shape

    pad = k_size//2#取余
    #防止卷积之后图片变小，
    out = np.zeros((H+pad*2,W+pad*2,C),dtype=np.float)
    #将图片放在最中间
    out[pad:pad+H,pad:pad+W]= img.copy().astype(np.float)


    K=np.zeros((k_size,k_size),dtype=np.float)
    #计算掩膜
    #根据高斯公式计算掩膜每一块的数值，并且进行归一化
    for x in range(-pad,-pad+k_size):
        for y in range(-pad,-pad+k_size):
            K[y+pad,x+pad]=np.exp(-(x**2+y**2)/(2*(sigma**2)))
    K /=(2*np.pi*sigma*sigma)
    K /=K.sum()

    tmp = out.copy()

    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y,pad+x,c]=np.sum(K*tmp[y:y+k_size,x:x+k_size,c])

    out = np.clip(out,0,255)
    out = out[pad:pad+H,pad:pad+W].astype(np.uint8)
    return out


img = cv2.imread('imori_noise.jpg')
out = guss_filter(img)


cv2.imshow('out',out)
cv2.waitKey(0)
cv2.destroyAllWindows()

