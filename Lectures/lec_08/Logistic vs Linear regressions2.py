print(__doc__)

# http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html
# Code source: Gael Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# this is our test set, it's just a straight line with some
# Gaussian noise
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
y = (X > 0).astype(np.float)
X[X > 0] *= 4
X += .3 * np.random.normal(size=n_samples)
X = X[:, np.newaxis]

# divide to test and train samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

# run the classifier
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X_train, y_train)

ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)

X_model = np.linspace(-5, 10, 300)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred, normalize=False))


# plot train results
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X_train.ravel(), y_train, color='black', zorder=20)

print("Coefficients", clf.coef_, "Intercept", clf.intercept_)

def model(x):
    return 1 / (1 + np.exp(-x))
loss = model(X_model * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_model, loss, color='red', linewidth=3)

plt.plot(X_model, ols.coef_ * X_model + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')

plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')

# plot test results
plt.figure(2, figsize=(4, 3))
plt.clf()
plt.scatter(X_test.ravel(), y_test, color='black', zorder=20)

print("Coefficients", clf.coef_, "Intercept", clf.intercept_)

def model(x):
    return 1 / (1 + np.exp(-x))
loss = model(X_model * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_model, loss, color='red', linewidth=3)

plt.plot(X_model, ols.coef_ * X_model + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')

plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')

plt.show()