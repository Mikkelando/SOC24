import numpy as np
from base_model import DynamicModel

class InfiniteFriedkinModel(DynamicModel):
    def __init__(self, delta, K, L1=100):
        """
        Инициализация модели Фридкина для бесконечного числа участников.
        
        :param delta: Параметр для генерации начальных мнений.
        :param K: Параметр для (2K+1)-диагональной матрицы влияния.
        :param L1: Максимальное число участников для отображения.
        """
        self.delta = delta
        self.K = K
        self.L1 = L1
        self.max_agents = 1000  # Установим максимальное число агентов для ограничения
        initial_state = self.generate_initial_state()
        influence_matrix = self.generate_influence_matrix()
        # stubbornness = self.generate_stubbornness()
       

        super().__init__(initial_state)
        self.influence_matrix = influence_matrix
        self.stubbornness = 0.0 * np.ones(len(self.influence_matrix))

    def generate_initial_state(self):
        """
        Генерация начального состояния.
        """
        return np.array([i * self.delta for i in range(1, self.L1 + 1)])

    def generate_influence_matrix(self):
        """
        Генерация (2K+1)-диагональной матрицы влияния.
        """
        a = np.random.rand(self.K + 1)
        W = np.zeros((self.L1, self.L1))
        
        for i in range(self.L1):
            for j in range(max(0, i - self.K), min(self.L1, i + self.K + 1)):
                W[i, j] = a[abs(i - j)]
        
        # Нормализация строк матрицы влияния
        W = W / W.sum(axis=1, keepdims=True)
        return W

    def generate_stubbornness(self, stubbornness):
        """
        Генерация вектора упрямства агентов.
        """
        self.stubbornness = stubbornness * np.ones(len(self.influence_matrix))
        return stubbornness * np.ones(len(self.influence_matrix))
        # return np.random.rand(self.L1)

    def step(self):
        """
        Выполняет один шаг динамики по модели Фридкина.
        """
        self.current_state = (np.eye(len(self.current_state)) - self.stubbornness) @ (self.influence_matrix @ self.current_state) + self.stubbornness @ self.initial_state

    def adjust_participants(self, T):
        """
        Регулировка числа участников для корректной динамики.
        """
        L = min(self.L1 + 10 * self.K * T, self.max_agents)
        self.L1 = L
        self.initial_state = self.generate_initial_state()
        self.influence_matrix = self.generate_influence_matrix()
        # self.stubbornness = self.generate_stubbornness()
        self.current_state = np.array(self.initial_state)
        self.states_cache = {0: np.array(self.initial_state)}
