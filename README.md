# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: pragadeesh M
RegisterNumber: 212225040309 
*/
import numpy as np

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
# Features: [Hours Studied, Attendance, Previous Marks]
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

# Target: Marks Scored
y = np.array([50, 45, 70, 80, 95], dtype=float)

# ------------------------------
# Step 2: Feature normalization
# ------------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]  # shape becomes (n_samples, n_features + 1)

# ------------------------------
# Step 3: Initialize weights
# ------------------------------
n_features = X.shape[1]
weights = np.zeros(n_features)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# ------------------------------
# Step 4: Stochastic Gradient Descent
# ------------------------------
for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi

print("Trained Weights (including intercept):", weights)

# ------------------------------
# Step 5: Make predictions
# ------------------------------
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)

```

## Output:
<img width="1397" height="119" alt="image" src="https://github.com/user-attachments/assets/837d07c0-af17-425a-b694-6f3f4874e86b" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
