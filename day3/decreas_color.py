# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 15:33
# @Author  : panfei
# @FileName: decreas_color.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

# 彩色图像由三通道像素组成，每个通道表示红、绿、蓝三原色中一种颜色的亮度值，每个数值都是 8 位的无符号字符类型（uchar），
# 因此颜色总数（number of colors，而是像素总数）为 ：
#
# 256×256×256=2^24=16777216
#
# 超过 1600 万种颜色，因此为了降低分析的复杂性，有时需要减少图像中颜色的数量，一种实现方法是把 RGB 空间细分到大小相等的方块中。
# 例如，如果把每种颜色数量减少到 1/8，那么颜色总数就变为 32×32×3232×32×32。
# 将旧图像中的每一个颜色值划分成一个方块，并将该方块的中间值作为新的颜色值。新图像使用新的颜色值，颜色数就减少了：

import cv2
img = cv2.imread('imori.jpg')

#//N*N+N/2
decreas_img = img//64*64+32

cv2.imshow('decreas_img',decreas_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
