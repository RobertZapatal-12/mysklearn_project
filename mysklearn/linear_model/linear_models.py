from abc import abstractmethod, ABC
import numpy as np

class LinearRegression:
    """Univariate linear regression model trained with gradient descent.

    The model represents the relationship between one input feature and a
    target value as ``y = m * x + b``. The slope and intercept are updated
    during each training epoch and are available through ``m_now`` and
    ``b_now``.

    Args:
        L (float): Learning rate used to update the model parameters.
        epoch (int): Number of gradient descent iterations performed by
            :meth:`fit`.
    """

    def __init__(self, L, epoch):
        self.L = L
        self.epoch = epoch
        self.m_now = 0
        self.b_now = 0

    @abstractmethod
    def fit(self, x, y ) :
        """Fit a univariate linear regression model using gradient descent.

        Args:
            x (np.ndarray or pd.DataFrame): Training feature values. This
                implementation expects one feature with shape
                (n_samples,) or (n_samples, 1).
            y (pd.Series or np.ndarray): Target values for each training
                sample, with shape (n_samples,).

        Returns:
            tuple[float, float]: The fitted slope and intercept as
            ``(m, b)``, where the model prediction is ``m * x + b``.

        The fitted values are also stored in ``self.m_now`` and
        ``self.b_now``.
        """
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



    
