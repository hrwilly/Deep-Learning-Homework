#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:35:00 2024

@author: agnesliang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#### a and b
def analyze_polynomial_regression(file_name, degrees):
    # read data and extract variables
    data = pd.read_csv(file_name, header=None)
    X_train = data.iloc[:, 0].values.reshape(-1, 1)  
    y_train = data.iloc[:, 1].values  
    X_val = data.iloc[:, 2].values.reshape(-1, 1) 
    y_val = data.iloc[:, 3].values  
    
    # initialize parameters and list to hold results
    train_mse = []
    val_mse = []

    # make copies to avoid overwriting
    X_train_original = X_train.copy()
    X_val_original = X_val.copy()

    for p in degrees:
        poly = PolynomialFeatures(degree=p, include_bias=False)
        
        # only fit_transform on Xtrain not Xval to avoid data leaking
        X_train_poly = poly.fit_transform(X_train_original)
        X_val_poly = poly.transform(X_val_original)
        
        # train linear regression
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # predict on training and validation dataset
        y_train_pred = model.predict(X_train_poly)
        y_val_pred = model.predict(X_val_poly)
        
        # calculate MSE 
        train_mse.append(mean_squared_error(y_train, y_train_pred))
        val_mse.append(mean_squared_error(y_val, y_val_pred))
    
    # find p that minimize MSE for training and validation dataset
    min_train_mse = min(train_mse)
    best_p_train = degrees[train_mse.index(min_train_mse)]

    min_val_mse = min(val_mse)
    best_p_val = degrees[val_mse.index(min_val_mse)]

    # plot mse against p
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_mse, label='Training MSE', marker='o')
    plt.plot(degrees, val_mse, label='Validation MSE', marker='o')
    plt.xlabel('Degree of Polynomial (p)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'MSE on Training and Validation Set vs Degree of Polynomial\nFile: {file_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # return the minimum MSE and corresponding p
    return min_train_mse, best_p_train, min_val_mse, best_p_val

# a. data1.csv
degrees = list(range(1,100))
train_mse_1, p_train_1, val_mse_1, p_val_1 = analyze_polynomial_regression('data1.csv', degrees)
print(f"(a). data1.csv -> Training MSE: {train_mse_1:.4f} at p = {p_train_1}, Validation MSE: {val_mse_1:.4f} at p = {p_val_1}")

# b. data2.csv
degree_2 = list(range(1, 100))
train_mse_2, p_train_2, val_mse_2, p_val_2 = analyze_polynomial_regression('data2.csv', degree_2)
print(f"(b). data2.csv -> Training MSE: {train_mse_2:.4f} at p = {p_train_2}, Validation MSE: {val_mse_2:.4f} at p = {p_val_2}")

# compare results 
print("""
Despite the training sets being identical in both datasets, the distribution of features or noise in the validation sets could have led to differences in generalization performance. The validation set in data2.csv performs best with lower-degree polynomial models, while higher-degree models quickly lead to increased validation errors. This suggests that the validation set in data2.csv is more sensitive to model complexity, where overly complex models decrease generalization capability rather than improving it.
""")




