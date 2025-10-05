import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Callable


class BaseKnn(ABC):
    """
    Базовый класс для всех KNN
    """

    def __init__(self, k: int = 3, distance_metric: str = "euclidean", weights: str = 'uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели <=> запоминание данных
        :param X: координаты меток
        :param y: метки классов
        :return:
        """
        self.X_train = X
        self.y_train = y

    def _calculate_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление расстояний между точками
        :param X: координаты меток
        :return:
        """
        if self.distance_metric == "euclidean":
            return
        elif self.distance_metric == "manhattan":
            return
        else:
            raise ValueError("Не существует такой метрики!")

    def _euclidean_distance(self, X: np.ndarray) -> np.ndarray:
        distance = np.sqrt((self.X_train - X[:, np.newaxis])** 2).sum(aixs=2)
        return distance

    def _manhattan_distance(self, X: np.ndarray) -> np.ndarray:
        distances = np.abs(self.X_train - X[:, np.newaxis]).sum(axis=2)
        return distances

    def _get_weights(self, distance: np.ndarray) -> np.ndarray:
        """
        Вычисление весов для взвешенного KNN
        :param distance:
        :return:
        """

        if self.weights == 'uniform':
            return np.ones_like(distance)
        elif self.weights == 'distance':
            with np.errstate(divide='ignore'):
                weights = 1.0 / (distance + 1e-8)
            weights[distance] = 1
            return weights
        else:
            raise ValueError("Не доступный тип вычисления весов")

    @abstractmethod
    def predict(self, X: np.ndarray):
        """Абстрактный метод для предсказания"""
        pass
