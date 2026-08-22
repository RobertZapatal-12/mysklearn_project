from abc import abstractmethod, ABC
import numpy as np

class LinearRegression:
    def __init__(self, L, epoch):
        self.L = L
        self.epoch = epoch
        self.m_now = 0
        self.b_now = 0

    def fit(self, x, y):
        for i in range(self.epoch):
            m_gradient = 0
            b_gradient = 0

            n = len(x)

            m_gradient = -(2/n) * np.sum(x*(y - self.m_now * x + self.b_now))
            b_gradient = -(2/n) * np.sum((y - self.m_now * x + self.b_now))

            self.m_now = self.m_now - self.L * m_gradient
            self.b_now = self.b_now - self.L * b_gradient
        print("The model has been fitted")
        return self.m_now, self.b_now



    
