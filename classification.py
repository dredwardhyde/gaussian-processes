from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import fetch_openml
import random

random.seed(1)

SAMPLE_SIZE = 2000

images, targets = fetch_openml('mnist_784', version=1, return_X_y=True)

X = images.to_numpy() / 255.0
y = targets.to_numpy()
shuffled = random.sample(range(len(X)), SAMPLE_SIZE)

X_train = [X[x] for x in shuffled]
Y_train = [y[x] for x in shuffled]

test_shuffled = set(range(len(X))) - set(shuffled)

X_test = [X[x] for x in test_shuffled]
y_test = [y[x] for x in test_shuffled]

gp = GaussianProcessClassifier(kernel=RBF(length_scale=1.0), optimizer=None,
                               multi_class='one_vs_rest', n_jobs=-1)

gp.fit(X_train, Y_train)

y_predicted = gp.predict(X_test[0:500])
print(y_predicted)
print(y_test[0:500])
print("Accuracy: %.3f" % (accuracy_score(y_test[0:500], y_predicted)))
