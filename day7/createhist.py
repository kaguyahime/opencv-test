# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 16:44
# @Author  : panfei
# @FileName: createhist.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime




import cv2

import matplotlib.pyplot as plt

img = cv2.imread('imori_dark.jpg')
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig('out.png')

plt.show()