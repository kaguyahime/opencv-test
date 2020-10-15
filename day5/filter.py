# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 16:44
# @Author  : panfei
# @FileName: filter.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime


#一些常用的滤波器api
#Motion Filter是对角线相同的算子进行滤波，不单独编码了
import cv2

if __name__ == '__main__':
    img = cv2.imread('imori_noise.jpg')
    img1 = cv2.medianBlur(img,3)
    img2 = cv2.blur(img,(3,3))
    img3 = cv2.bilateralFilter(img,9,20,20)
    img4 = cv2.GaussianBlur(img,(3,3),1.5)
    # cv2.imshow('img1',img1)
    # cv2.imshow('img2', img2)
    # cv2.imshow('img3',img3)
    cv2.imshow('img4',img4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
