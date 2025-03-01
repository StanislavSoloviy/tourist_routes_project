import numpy as np
import Model
from time import perf_counter


"""Глобальные константы"""
Q_TRAIN = 10000                 # Количество обучающих примеров
Q_TEST = 3000                  # Количество тестовых примеров
DATEBASE_NAME = 'Landscape_classifier'          # Имя базы данных для обучения
X = np.arange(-10, 10.01, 0.1)      # Диапазон входящих значений Х для функций
BATCH_SIZE = 32                     # количество тренировочных изображений для обработки перед обновлением параметров модели
IMG_SHAPE = 150           # размерность к которой будет приведено входное изображение
NETWORK_NAME = "CNN.keras"             # Имя нейросети для загрузки и сохранения
EPOCHS = 10                      # Количество эпох
forward_dict = {"mountain" : 0, "buildings" : 1, "glacier" : 2, "forest" : 3, "sea" : 4, "street" : 5}
backward_dict = {0: "гора", 1: "здания", 2: "ледник", 3: "лес", 4: "водоём", 5: "улица"}




if __name__ == '__main__':
    start = perf_counter()
    """Основной раздел программы"""
    model = Model.NeuralNetwork()
    """Создание и обучение нейросети. Закомментировать, если модель обучена"""
    """Аргумент sp выставляется для наложения фильтра соль/перец на входные изображения. SNR- процент шума"""
    # model.create()
    # model.train()
    print(f"Время на обучение модели {perf_counter() - start}")

    """Загрузка обученной модели"""
    model.load()
    """Просмотр архитектуры модели"""
    #model.get_arch()
    """Проверка аргумент sp выставляется для наложения фильтра соль/перец на входные изображения"""
    #model.check()
    model.predict("99.jpg")

