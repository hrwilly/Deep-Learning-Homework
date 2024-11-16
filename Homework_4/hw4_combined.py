'''
MF850: Deep Learning, Statistical Learning
Homework 4

Authors:

Jiali Liang (U46600644)
Hannah Willy (U19019406)
'''

import sympy as sp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
from abc import abstractmethod, abstractstaticmethod
from typing import Tuple
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def compute_last_state(yN, s):
    """
    Compute the last state (j = N), where all remaining probability budget should be assigned to q_N.

    Parameters:
    - yN: Observed count for the last category.
    - s: Remaining probability budget (typically 1).
    
    Returns:
    - Dictionary with V_j (log-likelihood at this step), q_j (allocated probability), remaining_budget.
    """
    qN = s  # Assign all remaining budget to q_N
    VN = yN * np.log(qN) if qN > 0 else float('-inf')  # Calculate log-likelihood for last step; handle log(0) case
    return {"V_j": float(VN), "q_j": float(qN), "remaining_budget": s}

def compute_optimal_q_and_V(y, V_N, s_initial):
    """
    Dynamic programming to maximize log-likelihood V_j from V_{N} to V_1.
    At each j, initialize q_j and s from the starting state, then find critical points 
    of the objective function within remaining budget s_j. Choose the optimal q_j as 
    the critical point maximizing V_j; fallback to initial allocation if no valid point.

    Parameters:
    - y: Array of observed counts y_i for each category.
    - V_N: Initial value for dynamic programming (starting from step N).
    - s_initial: Initial probability budget, set to 1.
    
    Output:
    - List of optimal V_j, q_j, and remaining s_j values for each step.
    """
    # initialize parameters
    K = np.sum(y)  
    results = []
    V_next = V_N  
    s_j = s_initial  # Start with  s = 1

    # Iterate backward from N to 1
    for j in range(len(y), 0, -1):
        y_j = y[j - 1]  # Get the observed count for step j
        initial_q_j = y_j / K * s_j  # Initial allocation based on the proportion of y_j
        q = sp.Symbol('q', positive=True)  
        
        # Define objective function 
        f = y_j * sp.log(q) + V_next  # Objective function 
        df_dq = sp.diff(f, q)  # Differentiate wrt q
        critical_points = sp.solve(df_dq, q)  # Solve for critical points
        
        # Filter critical points within the range [0, s_j] 
        q_j_candidates = [cp.evalf() for cp in critical_points if cp.is_real and cp > 0 and cp <= s_j]

        # Choose the best candidate that maximizes V_j; fallback to initial allocation otherwise
        if q_j_candidates:
            q_j = max(q_j_candidates, key=lambda q_val: y_j * np.log(q_val) if q_val > 0 else float('-inf'))
        else:
            q_j = initial_q_j  # Use initial allocation if no valid critical point

        # Calculate V_j based on the chosen q_j
        V_j = y_j * np.log(q_j) if q_j > 0 else float('-inf')
        s_j -= q_j  # Update remaining budget

        # Store results
        results.append({
            "step": f"V_{j}",
            "V_j": float(V_j),
            "q_j": float(q_j),
            "remaining_budget": s_j,
            "index": j
        })
        V_next = V_j  # update V_j 

    return results

def binary_accuracy(ypred, y):
    return sum(ypred.round() == y)/float(y.shape[0])


def sklearn_logreg(X_train, y_train, X_test, y_test):
    sk_logr = LogisticRegression(fit_intercept=False, penalty=None)
    sk_logr.fit(X_train, y_train)
    return binary_accuracy(sk_logr.predict(X_test), y_test)


class HW1Data():
    @abstractmethod
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def data_split(self, test_size=0.33) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class SkLearnGenerator(HW1Data):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    @abstractstaticmethod
    def _generator(n_samples) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def data(self):
        return type(self)._generator(self.n_samples)

    def data_split(self, test_size=0.33):
        X, y = self.data()
        return train_test_split(X, y, test_size=test_size)


class Make_classification(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_classification(n_samples, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)


class Make_moons(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_moons(n_samples, noise=0.05)


class Make_circles(SkLearnGenerator):
    def __init__(self, n_samples):
        super().__init__(n_samples)

    @staticmethod
    def _generator(n_samples):
        return make_circles(n_samples, factor=0.5, noise=0.05)


# 4.1 Sample inputs for N=6
y = np.array([5, 10, 15, 20, 25, 30])  
s_initial = 1.0

# 4.1 (a): Compute last step V_N and q_N
last_step_result = compute_last_state(y[-1], s_initial)
print("\n 4.1 (a) Result for last state V_N and q_N:")
print(f"V_N = {last_step_result['V_j']}, q_N = {last_step_result['q_j']}")

# 4.1 (b) and (c):
optimal_results = compute_optimal_q_and_V(y, last_step_result["V_j"], s_initial)

# Part (b): V_{N-1} 
V_N_minus_1 = optimal_results[-2]["V_j"]  
print("\n 4.1 (b) Result for V_{N-1}:")
print(f"V_{len(y) - 1} = {V_N_minus_1}")

# 4.1 (c): q_i results
print("\n 4.1 (c) Results for q_i values:")
for result in optimal_results:
    step_label = result["step"]
    q_j = result["q_j"]
    j = result["index"]
    print(f"q_{j} = {q_j}")

# 4.2 (a)
n_samples = 10000
data = Make_classification(n_samples)
X, y = data.data()
X_train, X_test, y_train, y_test = data.data_split()

#define the logistic regression
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
logit_model = LogisticRegressionModel(input_dim=X.shape[1])

#define the loss function and optimizer
logit_criterion = nn.BCELoss()
logit_optimizer = optim.SGD(logit_model.parameters(), lr=0.01)

#train model
def train_model(X_train, y_train, model, criterion, optimizer, num_epochs=1000):
    for epoch in range(num_epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'\n Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#test model
def test_model(X_train, y_train, X_test, y_test, model):
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        train_inputs = torch.tensor(X_train, dtype=torch.float32)

        test_outputs = model(test_inputs)
        train_outputs = model(train_inputs)

        test_predicted = (test_outputs >= 0.5).float().view(-1)
        train_predicted = (train_outputs >= 0.5).float().view(-1)

        test_accuracy = accuracy_score(y_test, test_predicted)
        train_accuracy = accuracy_score(y_train, train_predicted)
        
    return train_accuracy, test_accuracy

train_model(X_train, y_train, logit_model, logit_criterion, logit_optimizer)
train_accuracy, test_accuracy = test_model(X_train, y_train, X_test, y_test, logit_model)
print(f'\n 4.2 (a) Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

data = Make_classification(n_samples)
X, y = Make_classification(n_samples).data()
X_train, X_test, y_train, y_test = data.data_split()

# 4.2 (b)
#visualization
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01),
                            torch.arange(y_min, y_max, 0.01))
    grid = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
    with torch.no_grad():
        Z = model(grid.float()).view(xx.shape).numpy()
    
    #plot decision boundary
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['skyblue', 'sienna'])
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=1)
    
    #plot data points and classify then as groups
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='skyblue', edgecolor='k', s=40, label="Category 1")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='sienna', edgecolor='k', s=40, label="Category 2")
    
    plt.title(title)
    plt.legend()
    plt.show()

#Make_moons
data_moon = Make_moons(n_samples)
X_moon, y_moon = data_moon.data()
X_moon_train, X_moon_test, y_moon_train, y_moon_test = data_moon.data_split()

moon_model = LogisticRegressionModel(input_dim=X_moon.shape[1])
moon_criterion = nn.BCELoss()
moon_optimizer = optim.SGD(moon_model.parameters(), lr=0.1)

train_model(X_moon_train, y_moon_train, moon_model, moon_criterion, moon_optimizer)
train_moon_acc, test_moon_acc = test_model(X_moon_train, y_moon_train, X_moon_test, y_moon_test, moon_model)
print(f"\n 4.2 (b) Make_moons Train Accuracy: {train_moon_acc * 100:.2f}%")
print(f"Make_moons Test Accuracy: {test_moon_acc * 100:.2f}%")

#plot boundary decision
plot_decision_boundary(X_moon, y_moon, moon_model, "Decision Boundary for Make_moons")

#Make_circles
data_circle = Make_circles(n_samples)
X_circle, y_circle =  data_circle.data()
X_circle_train, X_circle_test, y_circle_train, y_circle_test = train_test_split(X_circle, y_circle, test_size=0.2)

circle_model = LogisticRegressionModel(input_dim=X_circle.shape[1])
circle_criterion = nn.BCELoss()
circle_optimizer = optim.SGD(circle_model.parameters(), lr=0.1)

train_model(X_circle_train, y_circle_train, circle_model, circle_criterion, circle_optimizer)
train_circle_acc, test_circle_acc = test_model(X_circle_train, y_circle_train, X_circle_test, y_circle_test, circle_model)
print(f"\n Make_circles Train Accuracy: {train_circle_acc * 100:.2f}%")
print(f"Make_circles Test Accuracy: {test_circle_acc * 100:.2f}%")

#plot boundary decision
plot_decision_boundary(X_circle, y_circle, circle_model, "Decision Boundary for Make_circles")

print("""\n We can notice that the logit model's performance is good when data is distributed in moon shape, since there is 
an obvious line that separate most of the two categories correctly. However, when data is distributed in circles, there 
is no such line that could seperate the two groups very well, so the performance of the model is worse than the former.""")

# 4.2(c)
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        #use RELU which is nonlinear
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

#make_moon data
nn_model_moon = Net(input_dim = X_moon.shape[1])
nn_criterion_moon = nn.BCELoss()
nn_optimizer_moon = torch.optim.Adam(nn_model_moon.parameters(), lr=0.001)

train_model(X_moon_train, y_moon_train, nn_model_moon, nn_criterion_moon, nn_optimizer_moon)
train_moon_acc, test_moon_acc = test_model(X_moon_train, y_moon_train, X_moon_test, y_moon_test, nn_model_moon)
print(f"\n 4.2 (c) Make_moons Train Accuracy: {train_moon_acc * 100:.2f}%")
print(f"Make_moons Test Accuracy: {test_moon_acc * 100:.2f}%")

# visualization
plot_decision_boundary(X_moon, y_moon, nn_model_moon, "Decision Boundary for Make_moons")

#make_circle data
nn_model_circle = Net(input_dim = X_circle.shape[1])
nn_criterion_circle = nn.BCELoss()
nn_optimizer_circle = torch.optim.Adam(nn_model_circle.parameters(), lr=0.001)

train_model(X_circle_train, y_circle_train, nn_model_circle, nn_criterion_circle, nn_optimizer_circle)
train_circle_acc, test_circle_acc = test_model(X_circle_train, y_circle_train, X_circle_test, y_circle_test, nn_model_circle)
print(f"\n Make_moons Train Accuracy: {train_circle_acc * 100:.2f}%")
print(f"Make_moons Test Accuracy: {test_circle_acc * 100:.2f}%")

# visualization
plot_decision_boundary(X_circle, y_circle, nn_model_circle, "Decision Boundary for Make_circles")

print('\n Neural network model makes much better prediction in classification, including make_moons data and make_circles data.')

