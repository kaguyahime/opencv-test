# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 14:21
# @Author  : panfei
# @FileName: number_classfile.py
# @Software: PyCharm
# @Cnblogs ：https://github.com/kaguyahime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras



def show_single_image(img_arr):#这里定义了一个函数用来查看图片
    img_arr = img_arr.reshape(28,28)#把输入进来的数据都变成28×28的这样才能展示数据
    plt.imshow(img_arr,cmap = "binary")#显示图片，具体的话可以看我matplotlib的教程
    plt.show()

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1.5)
    plt.show()

def predict_data(test_data):
    pred = model.predict(test_data.reshape(-1,28,28,1))
    return np.argmax(pred)


def look_image(data):
    plt.figure()
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    plt.imshow(data)



if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # 在这里我们调用这个方法后直接就可以把数据集分成训练集和测试集，
    x_train = x_train.reshape(-1, 28, 28, 1)
    # 在这里因为下面要通入模型时需要图片数据是四维数据，这里就把数据转成四维，最后的一是指图片通道数，因为我们的图片本来就是只有一个通道（RGB有三个通道）所以直接写成一
    x_test = x_test.reshape(-1, 28, 28, 1)
    print(x_train.shape)
    # 在这里读取各个数据集类型的形状，查看一下有没有问题
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    show_single_image(x_train[2])#这里我们输入的训练集的第一张图片，可以通过中括号里的数字看其他不同的数据
    x_train = x_train / 255.
    x_test = x_test / 255.
    model = keras.models.Sequential()  # 先生成一个模型框架
    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  padding='same',
                                  activation='selu',
                                  input_shape=(28, 28, 1)
                                  ))
    model.add(keras.layers.SeparableConv2D(filters=128,
                                           kernel_size=3,
                                           padding='same',
                                           activation='selu',
                                           ))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.SeparableConv2D(filters=256,
                                           kernel_size=3,
                                           padding='same',
                                           activation='selu',
                                           ))
    model.add(keras.layers.SeparableConv2D(filters=256,
                                           kernel_size=3,
                                           padding='same',
                                           activation='selu',
                                           ))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.SeparableConv2D(filters=512,
                                           kernel_size=3,
                                           padding='same',
                                           activation='selu',
                                           ))
    model.add(keras.layers.SeparableConv2D(filters=512,
                                           kernel_size=3,
                                           padding='same',
                                           activation='selu',
                                           ))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='selu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',  # 求解模型的方法
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=3,  # history用来接收训练过程中的一些参数数值 #训练的参数#训练5遍
                        validation_data=(x_test, y_test))  # 实时展示模型训练情况

    plot_learning_curves(history)  # （输入训练时的返回值）

    show_single_image(x_test[0])
    print("模型的预测结果是：", predict_data(x_test[0]))
    model.save('my_model.h5')

    import tensorflow as tf
    from tensorflow import keras
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    font = cv2.FONT_HERSHEY_SIMPLEX
    model = keras.models.load_model('my_model.h5')  # 读取网络
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_path = "number.png"  # 在这里写入图片路径




    image = cv2.imread(image_path)  # 读取图片
    image_ = cv2.resize(image, (250, 250), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)  # 灰度化处理
    img_w = cv2.Sobel(image, cv2.CV_16S, 0, 1)  # Sobel滤波，边缘检测
    img_h = cv2.Sobel(image, cv2.CV_16S, 1, 0)  # Sobel滤波，边缘检测
    img_w = cv2.convertScaleAbs(img_w)
    _, img_w = cv2.threshold(img_w, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    img_h = cv2.convertScaleAbs(img_h)
    _, img_h = cv2.threshold(img_h, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    image = img_w + img_h
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    temp_data = np.zeros((250, 10))
    image = np.concatenate((temp_data, image, temp_data), axis=1)
    temp_data = np.zeros((10, 270))
    image = np.concatenate((temp_data, image, temp_data), axis=0)
    image = cv2.convertScaleAbs(image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for _ in contours:
        x, y, w, h = cv2.boundingRect(_)
        if w * h < 100:
            continue
        img_model = image[y - 10:y + h + 10, x - 10:x + w + 10]
        img_model = cv2.resize(img_model, (28, 28), interpolation=cv2.INTER_AREA)
        img_model = img_model / 255
        predict = model.predict(img_model.reshape(-1, 28, 28, 1))
        if np.max(predict) > 0.5:
            data_predict = str(np.argmax(predict))
            image_z = cv2.rectangle(image_, (x - 10, y - 10), (x + w - 10, y + h - 10), (255, 0, 0), 1)
            image_z = cv2.putText(image_z, data_predict, (x + 10, y + 10), font, 0.7, (0, 0, 255), 1)
            look_image(image_z)
            save = cv2.imwrite("image_predict2.png", image_z)