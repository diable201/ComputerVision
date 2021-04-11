import sys
print(sys.version)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (20.0, 10.0)
# from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('grades.csv')
print(data.shape)
print(data.head())

score = data['score'].values
grade = data['grade'].values

# plot regression dataset
# plt.scatter(score,grade)
# plt.show()
m = len(score)

x0 = np.ones(m)
X = np.array([x0, score, np.sqrt(score)]).T
Y = np.array(grade)

st_time = time.time()
newB = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
print("Time",time.time() - st_time)

print(newB)

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

inital_cost = cost_function(X, Y, newB)
print("Final cost",inital_cost)


# draw linear regression
def linreg(x):
    return newB[0] + newB[1] * x + newB[2] * np.sqrt(x)

x = np.arange(0, 100, 1)
plt.scatter(score, grade)
plt.plot(x, linreg(x), 'r')
plt.show()