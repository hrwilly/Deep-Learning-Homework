'''
MF850: Deep Learning, Statistical Learning
Homework 5

Authors:

Jiali Liang (U46600644)
Hannah Willy (U19019406)
'''

'''
Note: I didn't play around with the parameters too much
so the better classifier is not doing better lol

Graphs look ugly because im using small size training data
if it's increased to 50 training data, we're not encouraging overfitting 
(test accuracy for basic model will get to 80%)

'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np
import random

def make_dataset(version=None, test=False):
    if test:
        random_state = None
    else:
        random_states = [27, 33, 38]
        if version is None:
            version = random.choice(range(len(random_states)))
            print(f"Dataset number: {version}")
        random_state = random_states[version]
    return sklearn.datasets.make_circles(factor=0.7, noise=0.1, random_state=random_state)

# (a) 
def build_model():
    """
    A basic feedforward neural network with 2 hidden layers.
    """
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(2, 256)  
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    return NeuralNetwork()

# (d) Better Classifier with complex architecture and regularization
def build_better_classifier():
    """
    Constructs an improved feedforward neural network designed to generalize better
    by adding:
    - Dropout for regularization to prevent overfitting.
    - Weight decay in optimizer for L2 regularization.
    """
    class BetterClassifier(nn.Module):
        def __init__(self):
            super(BetterClassifier, self).__init__()
            self.fc1 = nn.Linear(2, 256) 
            self.dropout1 = nn.Dropout(0.1)  # Dropout (10%)
            self.fc2 = nn.Linear(256, 128)  
            self.dropout2 = nn.Dropout(0.1)  
            self.fc3 = nn.Linear(128, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.dropout1(self.relu(self.fc1(x)))
            x = self.dropout2(self.relu(self.fc2(x)))
            x = self.sigmoid(self.fc3(x))
            return x

    return BetterClassifier()

# Training function with flexibility for learning rate scheduler and weight decay
def train_model(model, X_train, y_train, epochs=500, lr=0.01, use_scheduler=False, weight_decay=0.0):
    """
    Trains the given model using binary cross-entropy loss and Adam optimizer.
    Add a learning rate scheduler and weight decay to avoid overfit for the better classifier

    Returns:
        nn.Module: Trained model.
    """
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Weight decay only if specified

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate only if specified
        if use_scheduler:
            scheduler.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                y_pred_binary = (y_pred > 0.5).float()
                train_acc = (y_pred_binary == y_train).float().mean().item()
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Train Accuracy = {train_acc * 100:.2f}%")
    return model


# (b) 
def evaluate_and_plot(model, X, y, title):
    """
    Evaluates the model and plot scatter plots
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_pred_binary = (y_pred > 0.5).float()
    correct_indices = (y_pred_binary == y).squeeze()
    incorrect_indices = ~correct_indices

    plt.scatter(X[correct_indices, 0], X[correct_indices, 1], c='green', label='Correctly Classified')
    plt.scatter(X[incorrect_indices, 0], X[incorrect_indices, 1], c='red', label='Incorrectly Classified')
    plt.title(title)
    plt.legend()
    plt.show()

# (c)
def plot_decision_boundary(model, X, y, title):
    """
    Plots the decision boundary
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(grid_tensor).numpy().reshape(xx.shape)

    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="cool", edgecolor="k", s=20)
    plt.title(title)
    plt.show()

# Main function
def main():
    print("\nWe set Version = 1 for stable and faster convergence\n")

    # Generate datasets 
    X_full, y_full = make_dataset(version=1)  
    X_train, y_train = X_full[:30], y_full[:30]  # Use 30 samples for training 
    X_test, y_test = X_full[30:], y_full[30:]  
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Train base model
    base_model = build_model()
    base_model = train_model(base_model, X_train, y_train, epochs=500, lr=0.01, use_scheduler=False, weight_decay=0.0)

    # Evaluate base model
    with torch.no_grad():
        base_train_acc = (base_model(X_train) > 0.5).float().eq(y_train).float().mean().item()
        base_test_acc = (base_model(X_test) > 0.5).float().eq(y_test).float().mean().item()
    print(f"Base Model - Train Accuracy: {base_train_acc * 100:.2f}%, Test Accuracy: {base_test_acc * 100:.2f}%")

    # Train better model
    better_model = build_better_classifier()
    better_model = train_model(better_model, X_train, y_train, epochs=500, lr=0.01, use_scheduler=True, weight_decay=1e-4)

    # Evaluate better model
    with torch.no_grad():
        better_train_acc = (better_model(X_train) > 0.5).float().eq(y_train).float().mean().item()
        better_test_acc = (better_model(X_test) > 0.5).float().eq(y_test).float().mean().item()
    print(f"Better Model - Train Accuracy: {better_train_acc * 100:.2f}%, Test Accuracy: {better_test_acc * 100:.2f}%")

    # Plots for Base Model
    evaluate_and_plot(base_model, X_train, y_train, "Base Model - Train Scatter")
    evaluate_and_plot(base_model, X_test, y_test, "Base Model - Test Scatter")
    plot_decision_boundary(base_model, X_train.numpy(), y_train.numpy().squeeze(), "Base Model - Train Decision Boundary")
    plot_decision_boundary(base_model, X_test.numpy(), y_test.numpy().squeeze(), "Base Model - Test Decision Boundary")

    # Plots for Improved Model
    evaluate_and_plot(better_model, X_train, y_train, "Better Model - Train Scatter")
    evaluate_and_plot(better_model, X_test, y_test, "better Model - Test Scatter")
    plot_decision_boundary(better_model, X_train.numpy(), y_train.numpy().squeeze(), "better Model - Train Decision Boundary")
    plot_decision_boundary(better_model, X_test.numpy(), y_test.numpy().squeeze(), "better Model - Test Decision Boundary")

if __name__ == "__main__":
    main()
