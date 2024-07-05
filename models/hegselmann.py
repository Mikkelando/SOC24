# hegselmann.py
from base_model import DynamicModel
import numpy as np

class HegselmannKrauseModel(DynamicModel):
    def __init__(self, initial_state, epsilon):
        """
        Инициализация модели Хегсельмана-Краусса.
        
        :param initial_state: Начальное состояние модели.
        :param epsilon: Пороговое значение для учета мнений.
        """
        super().__init__(initial_state)
        self.epsilon = epsilon
        
    def step(self):
        """
        Выполняет один шаг динамики по модели Хегсельмана-Краусса.
        """
        new_state = np.copy(self.current_state)
        for i, state_i in enumerate(self.current_state):
            neighbors = [state_j for state_j in self.current_state if abs(state_j - state_i) <= self.epsilon]
            new_state[i] = np.mean(neighbors) if neighbors else state_i
        self.current_state = new_state