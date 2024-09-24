import scipy as sp
from functools import partial

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
              X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
              X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
              X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
              X_p[i] = 3
            else:
              X_p[i] = 4

        ll = qwk(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
              X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
              X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
              X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
              X_p[i] = 3
            else:
              X_p[i] = 4
        return X_p
    def coefficients(self):
        return self.coef_['x']
