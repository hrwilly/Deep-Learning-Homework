'''
MF850: Deep Learning, Statistical Learning
Homework 2: Mean Squared Error via Linear Regression and Weighted Least Squares

Authors:

Jiali Liang (U46600644)
Hannah Willy (U19019406)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import polars as pl


def obtain_data(file_name):
    '''
        This function obtains the training and validation
        data used for analysis.
    
        Parameters
        ----------
        file_name : String
                    Name of the csv file containing the data
    
        Returns
        -------
        X_train : numpy.ndarray
                  X training data
        y_train : numpy.ndarray
                  y training data
        X_val : numpy.ndarray
                  X validation data
        y_val : numpy.ndarray
                  y validation data              
    '''
    # Reading in the csv
    data = pd.read_csv(file_name, header=None)

    # Separating out the columns into their respective dataset
    X_train = data.iloc[:, 0].values  
    y_train = data.iloc[:, 1].values  
    X_val = data.iloc[:, 2].values
    y_val = data.iloc[:, 3].values 

    return X_train, y_train, X_val, y_val

def plotter(x, mse, file_name, xlab, title, double = False, legend = False):
    '''
        This function plots the MSE results for a given x, either degree
        of polynomial or bandwidth.
    
        Parameters
        ----------
        x : List
            The x data for the plot
        mse : List
              The mean squared error, used as y data
        file_name : String
                    Name of the csv file containing the data
        x_lab : String
                The label to place on the x-axis
        title : String
                The title of the plot
        double : Boolean (default value is False)
                 If True, the training data and the validation data
                 are both plotted on the same plot
        legend : Boolean (default value is False)
                 If True, a legend is included on the plot
    '''
    
    plt.figure(figsize=(10, 6))
    # Determining whether both the training and validation data
    # are to be plotted, or just validation. Parts a and b have both,
    # whereas parts d and e only plot the validation
    if double == True:
        plt.plot(x, mse[0], marker='o', label = 'Training MSE')
        plt.plot(x, mse[1], marker='o', label = 'Validation MSE')
    else: plt.plot(x, mse, marker='o')
    
    plt.xlabel(xlab)
    plt.ylabel('Mean Squared Error (MSE)')
    # Determining if the title should be labeled
    # Data 1 or Data 2 based on the data that was
    # loaded in
    if file_name == 'data1.csv': lab = 'Data 1'
    else: lab = 'Data 2'
    
    plt.suptitle(title)
    plt.title(lab)
    plt.grid(True)
    if legend == True: plt.legend()
    plt.show()

def poly_analysis(p, X_train, y_train, X_val, y_val):
    '''
        This function performs the polynomial analysis and regression
        for a given degree of polynomial. The output is the mean squared 
        error for both the training data and the validation data
    
        Parameters
        ----------
        p : Integer
            The degree of the polynomial of fit
        X_train : numpy.ndarray
                  X training data
        y_train : numpy.ndarray
                  y training data
        X_val : numpy.ndarray
                  X validation data
        y_val : numpy.ndarray
                  y validation data

        Returns
        -------
        train_mse : Float
                    The mean squared error of the training data
        val_mse : Float
                  The mean squared error of the validation data
    '''

    # Generates a matrix that consists of all polynomial degrees of the feature
    poly = PolynomialFeatures(degree=p, include_bias=False)
        
    # Only fit_transform on Xtrain not Xval to avoid data leaking
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
        
    # Train linear regression
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
        
    # Predict on training and validation dataset
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)
        
    # Calculate MSE 
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    return train_mse, val_mse

def analyze_polynomial_regression(file_name, degrees):
    '''
        This function performs the polynomial analysis and regression
        for a given degree of polynomial. The output is the mean squared 
        error for both the training data and the validation data
    
        Parameters
        ----------
        file_name : String
                    Name of the csv file containing the data
        degrees : List
                  The list of all degrees of the polynomial of fit

        Returns
        -------
        min_train_mse : Float
                        The local minimum of the MSE of the 
                        training data
        best_p_train : Integer
                       The degree of the polynomial of fit
                       where the local minimum occurs of the
                       training data
        min_val_mse : Float
                      The local minimum of the MSE of the 
                      validation data
        best_p_val : Integer
                     The degree of the polynomial of fit
                     where the local minimum occurs of the
                     validation data
    '''
    # Read data and extract variables
    X_train, y_train, X_val, y_val = obtain_data(file_name)

    # Reshape X so it is compatible with the linear regression
    X_train = X_train.reshape(-1, 1)
    X_val = X_val.reshape(-1, 1)
    
    # Initialize lists to hold results
    train_mse = []
    val_mse = []

    # Iterate over all degrees of fit
    for p in degrees:
        # Calculate and store MSE
        mse_train, mse_val = poly_analysis(p, X_train, y_train, X_val, y_val)
        train_mse.append(mse_train)
        val_mse.append(mse_val)
    
    # Find p that minimize MSE for training and validation dataset
    min_train_mse = min(train_mse)
    best_p_train = degrees[train_mse.index(min_train_mse)]

    min_val_mse = min(val_mse)
    best_p_val = degrees[val_mse.index(min_val_mse)]

    # Plot MSE against p
    xlab = 'Degree of Polynomial (p)'
    title = 'MSE on Training and Validation Set vs Degree of Polynomial'
    plotter(degrees, [train_mse, val_mse], file_name, xlab, title, True, True)

    # Return the minimum MSE and corresponding p
    return min_train_mse, best_p_train, min_val_mse, best_p_val

def weighted_least_squares(file_name, bandwidth):
    '''
        This function performs the weighted least squares analysis to
        find mean squared error of different bandwidths
    
        Parameters
        ----------
        file_name : String
                    Name of the csv file containing the data
        bandwidth : List
                  The list of all bandwidths

        Returns
        -------
        bandwidth[min_tau] : Float
                             The bandwidth where the minimum MSE
                             occurs
        np.min(mse) : Float
                      The minimum value of the MSE
    '''
    
    # Read data and extract variables
    X_train, y_train, X_val, y_val = obtain_data(file_name)

    # This variable will be used later on as the x values involved 
    # in the predictions
    X_analysis = pl.DataFrame(X_val).rename({'column_0' : 'X'})

    # Initialize lists to hold results
    mse = []

    # Adding an intercept to the training data
    X = np.array([[1.0, x] for x in X_train])

    # Iterating over all bandwidths
    for tau in bandwidth:
        # Calculating the weight matrix for all values of X
        # in the validation set
        W = [weights(X_train, x, tau) for x in X_val]
        # Finding the estimates for the intercept and slope
        # using the weights and the training data
        b = [beta(X, w, y_train) for w in W]
        # Putting the data into another format so that making the prediction
        # is easier
        b = pl.DataFrame(b).transpose().rename({'column_0' : 'b0', 'column_1' : 'b1'})
        estimates = b.hstack(X_analysis)
        # The predicted y is beta_0 + beta_1 * x
        pred = estimates.with_columns((pl.col('b0') + pl.col('b1') * pl.col('X')).alias('Predict'))['Predict'].to_list()
        # Calculating and storing the MSE
        mse.append(mean_squared_error(y_val, pred))

    # plot MSE against bandwidth
    xlab = 'Bandwidth ($\\tau$)'
    title = 'MSE on Validation Set: Weighted Least Squares'
    plotter(bandwidth, mse, file_name, xlab, title)

    # Find bandwidth that minimizes MSE
    min_tau = np.where(mse == np.min(mse))
    # Return the minimum MSE and corresponding bandwidth
    return bandwidth[min_tau], np.min(mse)

def beta(X, W, y):
    # Finds the estimate of beta from the formula derived in part c
    return np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)

def weights(Xi, x, tau):
    # Calculates the weights of the weightes least squares method
    return np.diag(np.exp(-(Xi - x)**2 / (2*tau)))


# a. data1.csv
degrees = list(range(1,100))
train_mse_1, p_train_1, val_mse_1, p_val_1 = analyze_polynomial_regression('data1.csv', degrees)
print(f"(a). data1.csv -> Training MSE: {train_mse_1:.4f} at p = {p_train_1}, Validation MSE: {val_mse_1:.4f} at p = {p_val_1}")

# b. data2.csv
train_mse_2, p_train_2, val_mse_2, p_val_2 = analyze_polynomial_regression('data2.csv', degrees)
print(f"(b). data2.csv -> Training MSE: {train_mse_2:.4f} at p = {p_train_2}, Validation MSE: {val_mse_2:.4f} at p = {p_val_2}")

# compare results 
print("""
Despite the training sets being identical in both datasets, the distribution of features or noise in the validation sets could have led to differences in generalization performance. The validation set in data2.csv performs best with lower-degree polynomial models, while higher-degree models quickly lead to increased validation errors. This suggests that the validation set in data2.csv is more sensitive to model complexity, where overly complex models decrease generalization capability rather than improving it.
""")

bandwidth = np.linspace(0.01, 5, 150)
tau1, mse1 = weighted_least_squares('data1.csv', bandwidth)
print('The minimum mean squared error occurs at tau = ' + str(round(tau1[0], 3)) + ' at a value of mse = ' + str(round(mse1, 3)))

tau2, mse2 = weighted_least_squares('data2.csv', bandwidth)
print('The minimum mean squared error occurs at tau = ' + str(round(tau2[0], 3)) + ' at a value of mse = ' + str(round(mse2, 3)))

print("""
Between data1 and data2, the mean squared error increases faster as the bandwidth increases for data2. Data2 also has a smaller optimal bandwidth than data1, but this minimum occurs at a higher mean squared error than the minimum for data1.
""")

