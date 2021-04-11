import sys
print(sys.version)

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv('grades.csv')
print(data.shape)
print(data.head())

score = data['score'].values
grade = data['grade'].values

print(np.power(score, 2))

# plot regression dataset
# plt.scatter(score,grade)
# plt.show()

m = len(score)
x0 = np.ones(m)
X = np.array([x0, score, np.power(score,2)]).T

# Initial Coefficients
B = np.array([0, 0, 0])
Y = np.array(grade)
alpha = 0.0001

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

inital_cost = cost_function(X, Y, B)
print("Initial cost",inital_cost)

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

# 100000 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 1000)

# New Values of B
print("Founded weights",newB)

# Final Cost of new B
print("Final cost",cost_history[-1])


# draw linear regression
# def linreg(x):
#     return newB[0] + newB[1] * x + newB[2] * np.sqrt(x)

# x = np.arange(0, 100, 1)
# plt.scatter(score, grade)
# plt.plot(x, linreg(x), 'r')
# plt.show()