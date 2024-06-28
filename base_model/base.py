import numpy as np

class DynamicModel:
    def __init__(self, initial_state):
        """
        Инициализация модели динамики.
        
        :param initial_state: Начальное состояние модели.
        """
        self.initial_state = np.array(initial_state)
        self.current_state = np.array(initial_state)
        self.states_cache = {0: np.array(initial_state)}
        
    def step(self):
        """
        Выполняет один шаг динамики.
        Этот метод должен быть реализован в подклассах.
        """
        raise NotImplementedError("Метод step() должен быть реализован в подклассах.")
        
    def generate_nth_state(self, n):
        """
        Генерирует n-ое состояние модели.
        
        :param n: Номер состояния, которое нужно сгенерировать.
        :return: Состояние модели на шаге n.
        """
        if n in self.states_cache:
            return self.states_cache[n]
        
        # Начнем с последнего закэшированного состояния
        last_cached_step = max(self.states_cache.keys())
        self.current_state = np.array(self.states_cache[last_cached_step])
        
        for i in range(last_cached_step + 1, n + 1):
            self.step()
            self.states_cache[i] = np.array(self.current_state)
        
        return self.current_state

    def generate_states_up_to_n(self, n):
        """
        Генерирует состояния модели от начального до n-ого.
        
        :param n: Номер последнего состояния, которое нужно сгенерировать.
        :return: Список состояний модели от начального до n-ого.
        """
        states = []
        for i in range(n + 1):
            state = self.generate_nth_state(i)
            states.append(state)
        return states
