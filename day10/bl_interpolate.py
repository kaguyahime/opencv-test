# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 14:31
# @Author  : panfei
# @FileName: bl_interpolate.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2
import numpy as np

def interpolate(img,ax,ay):
    H,W,C = img.shape
    aH = int(H*ay)
    aW = int(W*ax)

    y = np.arange(aH).repeat(aW).reshape(aW,-1)
    x = np.tile(np.arange(aW),(aH,1))

    y = y/ay
    x = x/ax

    ix = np.floor(x).astype(np.int8)
    iy = np.floor(y).astype(np.int8)

    ix = np.minimum(ix, W - 2)
    iy = np.minimum(iy, H - 2)

    dx = x - ix
    dy = y - iy

    dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
    dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)
    weight_x = dx
    weight_y = 1-dy

    test = weight_x*weight_y

    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out




if __name__ == '__main__':
    img = cv2.imread('imori.jpg')
    out = interpolate(img,2,2)
    cv2.imshow('out',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()