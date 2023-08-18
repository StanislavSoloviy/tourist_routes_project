import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import json
import main
import os
# from tensorflow.keras.applications import VGG16
from keras.layers import Input, Reshape, Conv2D, MaxPooling2D, UpSampling2D, InputLayer, Conv2DTranspose, Dense, Flatten, BatchNormalization, Dropout
from keras.models import Model


class NeuralNetwork:
    def __init__(self):
        self.__network_path = main.NETWORK_NAME
        self.__model = None
        self.__batch_size = main.BATCH_SIZE  # Размер батча
        self.__img_shape = main.IMG_SHAPE  # Разрешение
        self.__datebase_name = main.DATEBASE_NAME  # Относительный путь до базы данных
        self.__q_train = main.Q_TRAIN  # Количество обучающих примеров
        self.__q_test = main.Q_TEST  # Количество обучающих примеров
        self.__epochs = main.EPOCHS  # Кол-во эпох


    def image2array(self, filelist):
        """Преобразование изображений в векторы.
        sp- прогоняем изображенеи через фильтр соль/перец
        SNR- процент шума"""
        image_array = []
        for image in filelist:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.__img_shape, self.__img_shape))
            image_array.append(img)
        image_array = np.array(image_array)
        image_array = image_array.reshape(image_array.shape[0], self.__img_shape, self.__img_shape, 3)
        image_array = image_array.astype('float32')
        image_array = image_array / 255.0
        return np.array(image_array)

    def convert_img(self, path):
        """Преобразование изображений из базы данных аод вход в нейросеть"""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.__img_shape, self.__img_shape))
        img = img / 255.0
        img = np.array(img).reshape(self.__img_shape, self.__img_shape, 3)
        return img

    def __call__(self, *args, **kwargs):
        return self.__model(*args, **kwargs)

    def create(self):
        """Создание модели"""
        input_img = Input(shape=(self.__img_shape, self.__img_shape, 3))
        x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(6, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=x)
        # Компилирование модели
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Представление модели
        model.summary()
        self.__model = model

    def save(self):
        """Сохранение модели"""
        self.__model.save(self.__network_path)
        print("Модель сохранена")

    def load(self):
        """Загрузка модели"""
        try:
            self.__model = tf.keras.models.load_model(self.__network_path)
            print(f"Moдель {self.__network_path} загружена")
        except:
            print(f"Moдель {self.__network_path} не найдена")

    def train(self):
        """Обучение модели
        sp=True - на вход моделиподаём изображения через фильтр соль/перец. На выходе ожидаем чистое изображение
        SNR- процент шума"""
        with open("train_x.txt", "r", encoding="utf-8") as file:
            x_train_path = [self.__datebase_name + '/training/' + row.strip() for row in file.readlines()]
        with open("train_y.txt", "r", encoding="utf-8") as file:
            y_train = [int(row) for row in file.readlines()]
        x_train = self.image2array(x_train_path[:min(self.__q_train, len(x_train_path))])
        y_train = tf.keras.utils.to_categorical(y_train[:min(self.__q_train, len(y_train))], 6)

        # Обучение модели
        history = self.__model.fit(x_train, y_train, batch_size=self.__batch_size,
                                   epochs=self.__epochs,
                                   validation_split=0.2)

        # Тестирование модели
        with open("test_x.txt", "r", encoding="utf-8") as file:
            x_test_path = [self.__datebase_name + '/testing/' + row.strip() for row in file.readlines()]
        with open("test_y.txt", "r", encoding="utf-8") as file:
            y_test = [int(row) for row in file.readlines()]
        x_test = self.image2array(x_test_path[:self.__q_test])
        y_test = tf.keras.utils.to_categorical(y_test[:self.__q_test], 6)

        self.__model.evaluate(x_test, y_test)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Построение графика точности
        epochs_range = range(self.__epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Точность на обучении')
        plt.plot(epochs_range, val_acc, label='Точность на валидации')
        plt.legend(loc='lower right')
        plt.title('Точность на обучающих и валидационных данных')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Потери на обучении')
        plt.plot(epochs_range, val_loss, label='Потери на валидации')
        plt.legend(loc='upper right')
        plt.title('Потери на обучающих и валидационных данных')
        plt.savefig('./foo.png')
        plt.show()
        print("Модель обучена")
        self.save()

    def get_arch(self):
        if self.__model:
            self.__model.summary()
        else:
            try:
                self.load()
                self.__model.summary()
            except:
                self.create()

    def predict(self, img_way, show_img=False):
        """Предсказание изображения"""
        img = self.convert_img(img_way)
        x = np.expand_dims(img, axis=0)  # представляем изображение в виде трёхмерного тензора
        res = self.__model.predict(x)
        print(res)
        print(f"Ожидаемое значение: {main.backward_dict[np.argmax(res)]}")  # Аргумент с наибольшей вероятностью
        if show_img:
            plt.imshow(img)
            plt.show()


    @staticmethod
    def plot_digits(*args):
        args = [x.squeeze() for x in args]
        n = min([x.shape[0] for x in args])

        plt.figure(figsize=(2 * n, 2 * len(args)))
        for j in range(n):
            for i in range(len(args)):
                ax = plt.subplot(len(args), n, i * n + j + 1)
                plt.imshow(args[i][j])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.show()


