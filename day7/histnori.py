# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 16:44
# @Author  : panfei
# @FileName: histnori.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime


import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_normalization(img,a = 0,b = 255):
    c = img.min()
    d = img.max()

    out = img.copy()

    out = (b-a)/(d-c)*(out-c)+a

    out[out<a]=a
    out[out>b]=b

    out = out.astype(np.uint8)
    return out


if __name__ == '__main__':
    img = cv2.imread('imori_dark.jpg').astype(np.float)
    out = hist_normalization(img)

    plt.hist(out.ravel(),bins=255,rwidth=0.8,range=(0,255))
    plt.savefig('outhist.png')
    plt.show()

    cv2.imshow('out',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()