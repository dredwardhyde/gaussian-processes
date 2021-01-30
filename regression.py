import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)

# ============================================= TARGET FUNCTION ========================================================
def f(x):
    return x * np.sin(x)


# ============================================= GAUSSIAN KERNELS =======================================================
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

# ============================================== WITHOUT NOISE =========================================================
X = np.reshape([1., 3., 5., 6., 7., 8.], (1, -1)).T
y = f(X).reshape(-1)


gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X, y)

x = np.reshape(np.linspace(0, 10, 1000), (1, -1)).T
y_pred, sigma = gp.predict(x, return_std=True)

# ============================================== PLOT RESULTS ==========================================================
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                         (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')


# ================================================ WITH NOISE ==========================================================
X = np.reshape(np.linspace(0.1, 9.9, 20), (1, -1)).T
y = f(X).reshape(-1)
dy = 0.5 + 1.0 * np.random.random(y.shape)
y += np.random.normal(0, dy)

gp = GaussianProcessRegressor(kernel=kernel, alpha=dy, n_restarts_optimizer=10)
gp.fit(X, y)

y_pred, sigma = gp.predict(x, return_std=True)

# ============================================== PLOT RESULTS ==========================================================
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.errorbar(X, y, dy, fmt='r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                         (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()
