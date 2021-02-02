import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Formula 9.28
# Evaluation of log likelihood
def gmm_log_likelihood(X, means, covs, mixing_coefs):
    sum2 = 0
    for i in range(X.shape[0]):
        sum1 = 0
        for k in range(mixing_coefs.shape[0]):
            sum1 += mixing_coefs[k] * multivariate_normal.pdf(X[i], mean=means[k], cov=covs[k])
        sum2 += np.log(sum1)
    log_likelihood = sum2
    return log_likelihood


# Formula 9.23
# Calculating the responsibilities of Gaussians for creating the given training data point
def e_step(X, means, covs, mixing_coefs):
    responsibilities = np.zeros((X.shape[0], means.shape[0]))
    num = np.zeros(mixing_coefs.shape[0])
    for n in range(X.shape[0]):
        denomin = 0
        for k in range(mixing_coefs.shape[0]):
            num[k] = mixing_coefs[k] * multivariate_normal.pdf(X[n, :], means[k, :], covs[k, :, :])
            denomin += mixing_coefs[k] * multivariate_normal.pdf(X[n, :], means[k, :], covs[k, :, :])
        for k in range(mixing_coefs.shape[0]):
            responsibilities[n, k] = num[k] / denomin
    return responsibilities


# Given the responsibilities, we update the parameters of Gaussians - the means,
# covariance matrices and mixing coeeficients
def m_step(X, responsibilities):
    N = np.zeros(responsibilities.shape[1])
    # Iterate over clusters
    for k in range(responsibilities.shape[1]):
        # Formula 9.27
        N[k] = responsibilities[:, k].sum(axis=0)
        sum_mean = 0
        sum_cov = 0
        # Formula 9.24
        # New means
        for i in range(X.shape[0]):
            sum_mean += responsibilities[i, k] * X[i, :]
        sum_mean /= N[k]
        # Formula 9.25
        # New covariances
        for i in range(X.shape[0]):
            A = X[i, :] - sum_mean
            sum_cov += responsibilities[i, k] * np.outer(A, A.T)
        means[k, :] = sum_mean
        covs[k, :, :] = sum_cov / N[k]
    # Formula 9.26
    # New mixing coefficients
    mixing_coefs = N / N.sum(axis=0)
    return means, covs, mixing_coefs


def plot_gmm_2d(X, responsibilities, means, covs, mixing_coefs):
    plt.figure(figsize=[6, 6])
    palette = [[1, 0, 0], [0, 0, 1]]
    colors = responsibilities.dot(palette)
    # Plot the samples colored according to p(z|x)
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
    # Plot locations of the means
    for ix, m in enumerate(means):
        plt.scatter(m[0], m[1], s=150, marker='X', color=palette[ix], edgecolors='k', linewidths=1, )
    # Plot contours of the Gaussian
    x, y = np.mgrid[0:1:0.02, 0:1:0.02]
    pos = np.dstack((x, y))
    for k in range(len(mixing_coefs)):
        rv = multivariate_normal([means[k][0], means[k][1]], [[covs[k][0, 0], covs[k][0, 1]],
                                                              [covs[k][0, 1], covs[k][1, 1]]])
        plt.contour(x, y, rv.pdf(pos), 2, colors='k')
    plt.xlim(0, 1)
    plt.ylim(0, 1)


X = np.loadtxt('Old Faithful geyser.txt')
X_norm = np.zeros(X.shape)
X_norm[:, 0] = (X[:, 0] - np.amin(X[:, 0])) / (np.amax(X[:, 0]) - np.amin(X[:, 0]))
X_norm[:, 1] = (X[:, 1] - np.amin(X[:, 1])) / (np.amax(X[:, 1]) - np.amin(X[:, 1]))
max_iters = 20

# Initialize parameters
means = np.array([[0.2, 0.6], [0.8, 0.4]])
covs = np.array([0.5 * np.eye(2), 0.5 * np.eye(2)])
mixing_coefs = np.array([0.5, 0.5])
prev_log_likelihood = gmm_log_likelihood(X_norm, means, covs, mixing_coefs)
print('Initial log likelihood {0}'.format(prev_log_likelihood))
responsibilities = e_step(X_norm, means, covs, mixing_coefs)
plot_gmm_2d(X_norm, responsibilities, means, covs, mixing_coefs)

# Perform the EM iterations
for i in range(max_iters):
    responsibilities = e_step(X_norm, means, covs, mixing_coefs)
    means, covs, mixing_coefs = m_step(X_norm, responsibilities)
    current_log_likelihood = gmm_log_likelihood(X_norm, means, covs, mixing_coefs)
    print('Iteration {0}\n log likelihood = {1:.2f}, improvement = {2:.2f}'
          .format(i, current_log_likelihood, current_log_likelihood - prev_log_likelihood))
    prev_log_likelihood = current_log_likelihood
plot_gmm_2d(X_norm, responsibilities, means, covs, mixing_coefs)
plt.show()
