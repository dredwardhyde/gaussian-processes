import functools

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd


def probabilistic_pca(data_dim, latent_dim, num_datapoints, stddv_datapoints):
    w = yield tfd.Normal(loc=tf.zeros([data_dim, latent_dim]),
                         scale=2.0 * tf.ones([data_dim, latent_dim]),
                         name="w")
    z = yield tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                         scale=tf.ones([latent_dim, num_datapoints]),
                         name="z")
    x = yield tfd.Normal(loc=tf.matmul(w, z),
                         scale=stddv_datapoints,
                         name="x")


num_datapoints = 5000
data_dim = 2
latent_dim = 1
stddv_datapoints = 0.5

concrete_ppca_model = functools.partial(probabilistic_pca,
                                        data_dim=data_dim,
                                        latent_dim=latent_dim,
                                        num_datapoints=num_datapoints,
                                        stddv_datapoints=stddv_datapoints)
model = tfd.JointDistributionCoroutineAutoBatched(concrete_ppca_model)
actual_w, actual_z, x_train = model.sample()
qw_mean = tf.Variable(tf.random.normal([data_dim, latent_dim]))
qz_mean = tf.Variable(tf.random.normal([latent_dim, num_datapoints]))
qw_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([data_dim, latent_dim]), bijector=tfb.Softplus())
qz_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([latent_dim, num_datapoints]), bijector=tfb.Softplus())


def factored_normal_variational_model():
    qw = yield tfd.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
    qz = yield tfd.Normal(loc=qz_mean, scale=qz_stddv, name="qz")


surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(factored_normal_variational_model)
w = tf.Variable(tf.random.normal([data_dim, latent_dim]))
z = tf.Variable(tf.random.normal([latent_dim, num_datapoints]))

target_log_prob_fn = lambda w, z: model.log_prob((w, z, x_train))
losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior=surrogate_posterior,
    optimizer=tf.optimizers.Adam(learning_rate=0.05),
    num_steps=200)
posterior_samples = surrogate_posterior.sample(50)
_, _, x_generated = model.sample(value=posterior_samples)

x_generated = tf.reshape(tf.transpose(x_generated, [1, 0, 2]), (2, -1))[:, ::47]

plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1, label='Actual data')
plt.scatter(x_generated[0, :], x_generated[1, :], color='red', alpha=0.1, label='Simulated data (VI)')
plt.legend()
plt.axis([-20, 20, -20, 20])
plt.show()
