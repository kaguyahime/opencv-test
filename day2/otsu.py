# -*- coding: utf-8 -*-
# @Time    : 2020/10/12 10:00
# @Author  : panfei
# @FileName: otsu.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime


import cv2

#可直接使用OTSU API
def otsu(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th,dist = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(th)
    print(th3)
    return dist,th3

# def otsu_code(img):
#     w,h,c= img.shape()
#     for i in range(0,255):



img = cv2.imread('imori.jpg')
out,ret3= otsu(img)

#cv2.imshow('out',out)
cv2.imshow('ret3',ret3)
cv2.waitKey(0)
cv2.destroyAllWindows()