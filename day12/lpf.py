# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 9:53
# @Author  : panfei
# @FileName: lpf.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime



import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("../huahua3.png",0)
    # 0.转化为灰度
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

    # 1.FFT快速傅里叶变换: 空域-->频域
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 2.中心化: 将低频移动到图像中心
    fftshift = np.fft.fftshift(dft)

    # 获取振幅谱(展示图片用): 20 * numpy.log()是为了将值限制在[0， 255]
    magnitude_spectrum = 20 * np.log(cv2.magnitude(fftshift[:, :, 0], fftshift[:, :, 1]))
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    f = fftshift * mask

    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    # 显示低通滤波处理图像
    plt.imshow(res, 'gray'), plt.title('High frequency graph')
    plt.axis('off')
    plt.show()


