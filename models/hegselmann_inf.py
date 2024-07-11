import numpy as np
from base_model import DynamicModel

class InfiniteHegselmannKrauseModel(DynamicModel):
    def __init__(self, delta, epsilon, L1=100):
        """
        Инициализация модели Хегсельмана-Краусса для бесконечного числа участников.
        
        :param delta: Параметр для генерации начальных мнений.
        :param epsilon: Пороговое значение для учета мнений.
        :param L1: Максимальное число участников для отображения.
        """
        self.delta = delta
        self.epsilon = epsilon
        self.L1 = L1
        self.max_agents = 1000  # Установим максимальное число агентов для ограничения
        initial_state = self.generate_initial_state()
        super().__init__(initial_state)

    def generate_initial_state(self):
        """
        Генерация начального состояния.
        """
        return np.array([i * self.delta for i in range(1, self.L1 + 1)])

    def step(self):
        """
        Выполняет один шаг динамики по модели Хегсельмана-Краусса.
        """
        new_state = np.copy(self.current_state)
        for i, state_i in enumerate(self.current_state):
            neighbors = [state_j for state_j in self.current_state if abs(state_j - state_i) <= self.epsilon]
            new_state[i] = np.mean(neighbors) if neighbors else state_i
        self.current_state = new_state

    def adjust_participants(self, T):
        """
        Регулировка числа участников для корректной динамики.
        """
        L = min(self.L1 + 10 * T, self.max_agents)
        self.L1 = L
        self.initial_state = self.generate_initial_state()
        self.current_state = np.array(self.initial_state)
        self.states_cache = {0: np.array(self.initial_state)}
