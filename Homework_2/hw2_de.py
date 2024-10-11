def weighted_least_squares(file_name):
    
    data = pd.read_csv(file_name, header=None) 
    
    X_train = data.iloc[:, 0].values
    y_train = data.iloc[:, 1].values 
    X_val = data.iloc[:, 2].values
    y_val = data.iloc[:, 3].values
    
    bandwidth = np.linspace(0.01, 5, 150)

    X_analysis = pl.DataFrame(X_val).rename({'column_0' : 'X'})

    mse = []
    X = np.array([[1.0, x] for x in X_train])
    
    for tau in bandwidth:
        W = [weights(X_train, x, tau) for x in X_val]
        b = [beta(X, w, y_train) for w in W]
        b = pl.DataFrame(b).transpose().rename({'column_0' : 'b0', 'column_1' : 'b1'})
        estimates = b.hstack(X_analysis)
        pred = estimates.with_columns((pl.col('b0') + pl.col('b1') * pl.col('X')).alias('Predict'))['Predict'].to_list()
        mse.append(mean_squared_error(y_val, pred))

    plotter(bandwidth, mse, file_name)
    
    min_tau = np.where(mse == np.min(mse))
    return bandwidth[min_tau], np.min(mse)

def beta(X, W, y):
    return np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)

def weights(Xi, x, tau):
    return np.diag(np.exp(-(Xi - x)**2 / (2*tau)))

def plotter(bandwidth, mse, file_name):
    
    plt.figure(figsize=(10, 6))
    plt.plot(bandwidth, mse, marker='o')
    plt.xlabel('Bandwidth ($\\tau$)')
    plt.ylabel('Mean Squared Error (MSE)')
    if file_name == 'data1.csv': lab = 'Data 1'
    else: lab = 'Data 2'
    plt.suptitle(f'MSE on Validation Set: Weighted Least Squares')
    plt.title(lab)
    plt.grid(True)
    plt.show()

tau1, mse1 = weighted_least_squares('data1.csv')
print('The minimum mean squared error occurs at tau = ' + str(round(tau1[0], 3)) + ' at a value of mse = ' + str(round(mse1, 3)))

tau2, mse2 = weighted_least_squares('data2.csv')
print('The minimum mean squared error occurs at tau = ' + str(round(tau2[0], 3)) + ' at a value of mse = ' + str(round(mse2, 3)))

print('The optimal tau decreased, and the mse value increased between data1 and data2.')