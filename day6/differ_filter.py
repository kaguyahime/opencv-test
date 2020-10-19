# -*- coding: utf-8 -*-
# @Time    : 2020/10/19 8:57
# @Author  : panfei
# @FileName: differ_filter.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import cv2
import numpy as np

def different_filter(img,k_size = 3):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #补零操作
    W,H,C = img.shape
    pad = k_size//2
    out = np.zeros((H+pad*2,W+pad*2),dtype=np.float)
    out[pad:pad+H,pad:pad+W] = img_gray.copy().astype(np.float)

    tmp = out.copy()

    out_v = out.copy()
    out_h = out.copy()

    kv = [[0,0,0],[-1,1,0],[0,0,0]]
    kh = [[0,-1,0],[0,1,0],[0,0,0]]

    for y in range(H):
        for x in range(W):
            out_v[y:y+pad,x:x+pad] = np.sum(kv*tmp[y:y+k_size,x:x+k_size])
            out_h[y:y + pad, x:x + pad] = np.sum(kh * tmp[y:y + k_size, x:x + k_size])

    out_v = np.clip(out_v,0,255)
    out_h = np.clip(out_h,0,255)

    out_v = out_v[pad: pad + H, pad: pad + W].astype(np.uint8)
    out_h = out_h[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out_v,out_h

if __name__ == '__main__':
    img = cv2.imread('imori.jpg')
    out_v,out_h = different_filter(img)

    cv2.imwrite("out_v.jpg", out_v)
    cv2.imshow("result_v", out_v)
    while cv2.waitKey(100) != 27:  # loop if not get ESC
        if cv2.getWindowProperty('result_v', cv2.WND_PROP_VISIBLE) <= 0:
            break
    cv2.destroyWindow('result_v')

    cv2.imwrite("out_h.jpg", out_h)
    cv2.imshow("result_h", out_h)
    # loop if not get ESC or click x
    while cv2.waitKey(100) != 27:
        if cv2.getWindowProperty('result_h', cv2.WND_PROP_VISIBLE) <= 0:
            break
    cv2.destroyWindow('result_h')
    cv2.destroyAllWindows()







