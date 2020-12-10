# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 15:18
# @Author  : panfei
# @FileName: affine.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2
import numpy as np
# def affine(img, a, b, c, d, tx, ty):
#     H, W, C = img.shape
#
#     # temporary image
#     img = np.zeros((H+2, W+2, C), dtype=np.float32)
#     img[1:H+1, 1:W+1] = _img
#
#     # get new image shape
#     H_new = np.round(H * d).astype(np.int)
#     W_new = np.round(W * a).astype(np.int)
#     out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)
#
#     # get position of new image
#     x_new = np.tile(np.arange(W_new), (H_new, 1))
#     y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
#
#     # get position of original image by affine
#     adbc = a * d - b * c
#     x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
#     y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1
#
#     x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
#     y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)
#
#     # assgin pixcel to new image
#     out[y_new, x_new] = img[y, x]
#
#     out = out[:H_new, :W_new]
#     out = out.astype(np.uint8)
#
#     return out
#
#
# # Read image
# _img = cv2.imread("../imori.jpg").astype(np.float32)
# img = _img.copy()
# # Affine
# out = affine(img, a=1.3, b=0, c=0, d=0.8, tx=30, ty=-30)
#
#
# # Save result
# cv2.imshow("result", out)
# cv2.waitKey(0)
# cv2.imwrite("out.jpg", out)



import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('../imori.jpg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
rows,cols,ch=img2.shape
pts1=np.float32([[50,50],[200,50],[50,200]])
pts2=np.float32([[10,100],[200,50],[100,250]])
M=cv2.getAffineTransform(pts1,pts2)
dst=cv2.warpAffine(img2,M,(int(cols*2),int(rows*2)))
plt.subplot(121)
plt.imshow(img2),plt.title('Input')
plt.subplot(122)
plt.imshow(dst),plt.title('Output')
plt.show()
