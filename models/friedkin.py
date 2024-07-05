# friedkin.py
from base_model import DynamicModel
import numpy as np

class FriedkinModel(DynamicModel):
    def __init__(self, initial_state, influence_matrix, stubbornness):
        """
        Инициализация модели Фридкина.
        
        :param initial_state: Начальное состояние модели.
        :param influence_matrix: Матрица влияния (весов).
        :param stubbornness: Вектор упрямства агентов (значения от 0 до 1).
        """
        super().__init__(initial_state)
        self.influence_matrix = np.array(influence_matrix)
        self.stubbornness = np.array(stubbornness)
        
    def step(self):
        """
        Выполняет один шаг динамики по модели Фридкина.
        """
        self.current_state = (1 - self.stubbornness) * (self.influence_matrix @ self.current_state) + self.stubbornness * self.initial_state