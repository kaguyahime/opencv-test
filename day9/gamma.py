# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 15:51
# @Author  : panfei
# @FileName: gamma.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime


import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('imori_gamma.jpg').astype(np.float)
    out = img/255
    out = out**(1/2.2)
    out = out*255

    out = out.astype(np.uint8)

    cv2.imshow('out',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()