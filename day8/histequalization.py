# -*- coding: utf-8 -*-
# @Time    : 2020/11/8 14:22
# @Author  : panfei
# @FileName: histequalization.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime


import cv2
import numpy as np
import matplotlib.pyplot as plt

def equal(img,m0=128,s0=52):
    m = np.mean(img)
    print(m)
    s = np.std(img)
    print(s)

    out = img.copy()

    #m0:操作之后的平均值，m0越大最终数字越大，亮度越大
    #s0：操作之后的标准差，将差距扩大或缩小到规定标准差，s0越大，对比度越大

    out = s0/s*(out-m)+m0
    out[out<0]=0
    out[out>255]=255

    print(np.mean(out))
    print(np.std(out))

    out = out.astype(np.uint8)

    return out


def hist_normalization(img,a = 0,b = 255):
    #将图形中原来的只扩展到（0-255）之间使值的分布更加宽，对比更加明显
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
    out = equal(img)
    out1 = hist_normalization(img)
    plt.hist(out.ravel(),bins=255,rwidth=0.8,range=(0,255))
    plt.hist(out1.ravel(),bins=255,rwidth=1,range=(0,255))
    plt.hist(img.ravel(), bins=255, rwidth=1, range=(0, 255))
    plt.show()
    cv2.imshow('out', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
