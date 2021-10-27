import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import pandas as pd
import os

if __name__ == "__main__":

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
    x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
    y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
    # print(x_valid.shape, y_valid.shape)
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)

    def show_single_image(img_arr):
        plt.imshow(img_arr, cmap='binary')
        plt.show()

    # show_single_image(x_train[1])

    def show_imgs(n_rows, n_cols, x, y, class_names):
        assert len(x) == len(y)
        assert n_rows * n_cols < len(x)
        plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
        for row in range(n_rows):
            for col in range(n_cols):
                index = n_cols * row + col
                plt.subplot(n_rows, n_cols, index + 1)
                plt.imshow(x[index], cmap='binary', interpolation='nearest')
                plt.axis('off')
                plt.title(class_names[y[index]])
        plt.show()
    #
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # show_imgs(3, 5, x_train, y_train, class_names)
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    # 使用交叉熵损失函数，随机梯度下降法进行训练
    # 这里需要说明的是如果是categorical_crossentropy
    # 则需要将标签数据集进行one_hot处理，如果是sparse_categorical_crossentropy，
    # 则不需要
    model.compile(loss=tf.losses.sparse_categorical_crossentropy,
                  optimizer=optimizers.SGD(learning_rate=0.001),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
    print(model.summary())
    print(history.history)

    def plot_learning_curves(history):
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()
        plt.savefig('plot_learning_curves.jpg')

    plot_learning_curves(history)
