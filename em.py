import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
from scipy.stats import multivariate_normal


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    Xmu = X - mux
    Ymu = Y - muy

    rho = sigmaxy / (sigmax * sigmay)
    z = Xmu ** 2 / sigmax ** 2 + Ymu ** 2 / sigmay ** 2 - 2 * rho * Xmu * Ymu / (sigmax * sigmay)
    denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)
    return np.exp(-z / (2 * (1 - rho ** 2))) / denom


def normalize_data(X):
    X_norm = np.zeros(X.shape)
    X_norm[:, 0] = (X[:, 0] - np.amin(X[:, 0])) / (np.amax(X[:, 0]) - np.amin(X[:, 0]))
    X_norm[:, 1] = (X[:, 1] - np.amin(X[:, 1])) / (np.amax(X[:, 1]) - np.amin(X[:, 1]))
    return X_norm


def gmm_log_likelihood(X, means, covs, mixing_coefs):
    sum2 = 0
    for i in range(X.shape[0]):
        sum1 = 0
        for k in range(mixing_coefs.shape[0]):
            sum1 += mixing_coefs[k] * multivariate_normal.pdf(X[i], mean=means[k], cov=covs[k])
        sum2 += np.log(sum1)
    log_likelihood = sum2
    return log_likelihood


X = np.loadtxt('Old Faithful geyser.txt')
X_norm = normalize_data(X)

plt.figure(figsize=[6, 6])
plt.scatter(X_norm[:, 0], X_norm[:, 1]);
plt.xlabel('Eruptions (minutes)')
plt.ylabel('Waiting time (minutes)')
X_norm = normalize_data(X)
max_iters = 20

# Initialize the parameters
means = np.array([[0.2, 0.6], [0.8, 0.4]])
covs = np.array([0.5 * np.eye(2), 0.5 * np.eye(2)])
mixing_coefs = np.array([0.5, 0.5])

old_log_likelihood = gmm_log_likelihood(X_norm, means, covs, mixing_coefs)

print('At initialization: log-likelihood = {0}'
      .format(old_log_likelihood))


def e_step(X, means, covs, mixing_coefs):
    responsibilities = np.zeros((X.shape[0], means.shape[0]))
    num = np.zeros(mixing_coefs.shape[0])
    denomin = 0
    for n in range(X.shape[0]):
        denomin = 0
        for k in range(mixing_coefs.shape[0]):
            num[k] = mixing_coefs[k] * multivariate_normal.pdf(X[n, :], means[k, :], covs[k, :, :])
            denomin += mixing_coefs[k] * multivariate_normal.pdf(X[n, :], means[k, :], covs[k, :, :])
        for k in range(mixing_coefs.shape[0]):
            responsibilities[n, k] = num[k] / denomin
    return responsibilities


def m_step(X, responsibilities):
    N = np.zeros(responsibilities.shape[1])
    # iterate over clusters
    for k in range(responsibilities.shape[1]):
        # init
        N[k] = responsibilities[:, k].sum(axis=0)
        sum_mean = 0
        sum_cov = 0
        # new means
        for i in range(X.shape[0]):
            sum_mean += responsibilities[i, k] * X[i, :]
        sum_mean /= N[k]
        # new covs
        for i in range(X.shape[0]):
            A = X[i, :] - sum_mean
            sum_cov += responsibilities[i, k] * np.outer(A, A.T)
        means[k, :] = sum_mean
        covs[k, :, :] = sum_cov / N[k]
    NN = N.sum(axis=0)
    mixing_coefs = N / NN

    return means, covs, mixing_coefs


def plot_gmm_2d(X, responsibilities, means, covs, mixing_coefs):
    plt.figure(figsize=[6, 6])
    palette = np.array(sns.color_palette('colorblind', n_colors=3))[[0, 2]]
    colors = responsibilities.dot(palette)
    # Plot the samples colored according to p(z|x)
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
    # Plot locations of the means
    for ix, m in enumerate(means):
        plt.scatter(m[0], m[1], s=300, marker='X', c=palette[ix],
                    edgecolors='k', linewidths=1, )
    # Plot contours of the Gaussian
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x, y)
    for k in range(len(mixing_coefs)):
        zz = bivariate_normal(xx, yy, np.sqrt(covs[k][0, 0]),
                              np.sqrt(covs[k][1, 1]),
                              means[k][0], means[k][1], covs[k][0, 1])
        plt.contour(xx, yy, zz, 2, colors='k')
    plt.xlim(0, 1)
    plt.ylim(0, 1)


responsibilities = e_step(X_norm, means, covs, mixing_coefs)
print('At initialization: log-likelihood = {0}'
      .format(old_log_likelihood))
plot_gmm_2d(X_norm, responsibilities, means, covs, mixing_coefs)

# Perform the EM iteration
for i in range(max_iters):
    responsibilities = e_step(X_norm, means, covs, mixing_coefs)
    means, covs, mixing_coefs = m_step(X_norm, responsibilities)
    new_log_likelihood = gmm_log_likelihood(X_norm, means, covs, mixing_coefs)
    # Report & visualize the optimization progress
    print('Iteration {0}: log-likelihood = {1:.2f}, improvement = {2:.2f}'
          .format(i, new_log_likelihood, new_log_likelihood - old_log_likelihood))
    old_log_likelihood = new_log_likelihood
    plot_gmm_2d(X_norm, responsibilities, means, covs, mixing_coefs)

plt.show()