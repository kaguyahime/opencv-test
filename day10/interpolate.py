# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 10:53
# @Author  : panfei
# @FileName: interpolate.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2


if __name__ == '__main__':
    img = cv2.imread('imori.jpg')
    fx = 2
    fy = 2
    enlarge = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('enlarge',enlarge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()