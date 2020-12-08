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
    y = (y / ah).astype(np.int)
    x = (x / aw).astype(np.int)

    #[0,0,0]，[0,1,1]一维内加一个代表行内加一个，其中第一个数组中代表原图的行，第二个数组代表原图中的列
    #
    out = img[y,x].astype(np.uint8)
    out1 = img[y,x].astype(np.uint8)
    print(out)
    return out

if __name__ == '__main__':

    img = cv2.imread('imori.jpg').astype(np.float)
    out = interpolate(img,3,3)
    cv2.imshow('out',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # Nereset Neighbor interpolation
# def nn_interpolate(img, ax=1, ay=1):
#     H, W, C = img.shape
#
#     aH = int(ay * H)
#     aW = int(ax * W)
#
#     y = np.arange(aH).repeat(aW).reshape(aW, -1)
#     x = np.tile(np.arange(aW), (aH, 1))
#     y = np.round(y / ay).astype(np.int)
#     x = np.round(x / ax).astype(np.int)
#
#     out = img[y,x]
#
#     out = out.astype(np.uint8)
#
#     return out
#
#
# # Read image
# img = cv2.imread("imori.jpg").astype(np.float)
#
# # Nearest Neighbor
# out = nn_interpolate(img, ax=1.5, ay=1.5)
#
# # Save result
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.imwrite("out.jpg", out)





