from base_model import DynamicModel
import numpy as np

class DeGrootModel(DynamicModel):
    def __init__(self, initial_state, influence_matrix):
        """
        Инициализация модели Дегрута.
        
        :param initial_state: Начальное состояние модели.
        :param influence_matrix: Матрица влияния (весов).
        """
        super().__init__(initial_state)
        self.influence_matrix = np.array(influence_matrix)
        
    def step(self):
        """
        Выполняет один шаг динамики по модели Дегрута.
        """
        self.current_state = self.influence_matrix @ self.current_state
