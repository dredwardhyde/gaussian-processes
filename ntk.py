import os

os.environ['JAX_ENABLE_X64'] = 'True'
import jax.numpy as np

from jax import random
from jax.api import jit
import neural_tangents as nt
from neural_tangents import stax
import matplotlib.pyplot as plt

key = random.PRNGKey(10)
train_points = 20
test_points = 1000
noise_scale = 0.3

target_fn = lambda x: x * np.sin(x)
key, x_key, y_key = random.split(key, 3)
train_xs = np.reshape(np.linspace(0.1, 10, train_points), (1, -1)).T
train_ys = target_fn(train_xs)
train_ys += noise_scale * random.normal(y_key, (train_points, 1))
test_xs = x = np.reshape(np.linspace(0.1, 10, test_points), (1, -1)).T
test_xs = np.reshape(test_xs, (test_points, 1))
test_ys = target_fn(test_xs)

ResBlock = stax.serial(
    stax.FanOut(2),
    stax.parallel(
        stax.serial(
            stax.Erf(),
            stax.Dense(512, W_std=1.1, b_std=0),
        ),
        stax.Identity()
    ),
    stax.FanInSum()
)

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512, W_std=1, b_std=0),
    ResBlock, ResBlock, stax.Erf(),
    stax.Dense(1, W_std=1.5, b_std=0)
)

kernel_fn = jit(kernel_fn, static_argnums=(2,))
predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_xs, train_ys, diag_reg=1e-4)
ntk_mean, ntk_covariance = predict_fn(x_test=test_xs, get='ntk', compute_cov=True)
ntk_mean = np.reshape(ntk_mean, (-1,))
ntk_std = np.sqrt(np.diag(ntk_covariance))

plt.plot(train_xs, train_ys, 'r.', markersize=7, label='train')
plt.plot(test_xs, test_ys, 'r:', markersize=3, label='test')
plt.plot(test_xs, ntk_mean, 'b-', linewidth=0.5, label='Infinite Network')
plt.fill_between(
    np.reshape(test_xs, (-1)),
    ntk_mean - 1.96 * ntk_std,
    ntk_mean + 1.96 * ntk_std,
    alpha=.2, fc='g', ec='None', label='95% confidence interval')
plt.xlim(0, 10)
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()
