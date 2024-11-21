'''
MF850: Deep Learning, Statistical Learning
Homework 5

Authors:

Jiali Liang (U46600644)
Hannah Willy (U19019406)
'''

import torch
import torch.nn as nn
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

# (a) Base model
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

# Training function
def train_model(model, X_train, y_train, epochs=500, lr=0.01):
    """
    Trains the given model using binary cross-entropy loss and Adam optimizer.

    Returns:
        nn.Module: Trained model.
    """
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                y_pred_binary = (y_pred > 0.5).float()
                train_acc = (y_pred_binary == y_train).float().mean().item()
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Train Accuracy = {train_acc * 100:.2f}%")
    return model

# (b) Evaluation and plotting
def evaluate_and_plot(model, X, y, title):
    """
    Evaluates the model and plot scatter plots.
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

# (c) Decision boundary plotting
def plot_decision_boundary(model, X, y, title):
    """
    Plots the decision boundary.
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

# (d) Better classifier
def build_better_classifier(threshold=0.7):
    """
    Constructs a simplistic classifier that separates points based on their radial distance from the origin.
    This classifier uses a predefined circle to classify points inside the circle as Class 1 and outside as Class 0.

    Logic:
    - Computes the radial distance (sqrt(x^2 + y^2)) for each point.
    - Applies a threshold to determine the class based on distance from the origin.
    """

    class ScatterPlotClassifier(nn.Module):
        def __init__(self, threshold=0.7):
            super(ScatterPlotClassifier, self).__init__()
            self.threshold = threshold  # Radius of the circle for classification

        def forward(self, x):
            """
            Classifies points based on their distance from the origin.

            Args:
                x (torch.Tensor): Input features (2D points).

            Returns:
                torch.Tensor: Predicted labels (0 or 1).
            """
            # Calculate radial distance sqrt(x1^2 + x2^2) for each point
            distances = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
            # Classify based on threshold distance
            predictions = (distances < self.threshold).float().unsqueeze(1)  # Ensure output has shape [N, 1]
            return predictions

    return ScatterPlotClassifier(threshold)



if __name__ == "__main__":
    print("\nWe set Version = 1 for stable and faster convergence\n")

    # Generate datasets
    X_full, y_full = make_dataset(version=1)  
    X_train, y_train = X_full[:50], y_full[:50]  # Use 50 samples for training 
    X_test, y_test = X_full[50:], y_full[50:]  
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # NN model
    # Train model
    base_model = build_model()
    base_model = train_model(base_model, X_train, y_train, epochs=500, lr=0.01)

    # Evaluate model
    with torch.no_grad():
        base_train_acc = (base_model(X_train) > 0.5).float().eq(y_train).float().mean().item()
        base_test_acc = (base_model(X_test) > 0.5).float().eq(y_test).float().mean().item()
    print(f"Base Model - Train Accuracy: {base_train_acc * 100:.2f}%, Test Accuracy: {base_test_acc * 100:.2f}%")

    # Plots 
    evaluate_and_plot(base_model, X_train, y_train, "Base Model - Train Scatter")
    evaluate_and_plot(base_model, X_test, y_test, "Base Model - Test Scatter")
    plot_decision_boundary(base_model, X_train.numpy(), y_train.numpy().squeeze(), "Base Model - Train Decision Boundary")
    plot_decision_boundary(base_model, X_test.numpy(), y_test.numpy().squeeze(), "Base Model - Test Decision Boundary")

    # Better classifier
    # Train model
    print("\nEvaluating Better Classifier (Scatter Plot Logic)...")
    better_model = build_better_classifier(threshold=0.7)

    # Evaluate model
    better_train_predictions = better_model(X_train)
    better_test_predictions = better_model(X_test)
    better_train_acc = (better_train_predictions > 0.5).float().eq(y_train).float().mean().item()
    better_test_acc = (better_test_predictions > 0.5).float().eq(y_test).float().mean().item()
    print(f"Better Classifier - Train Accuracy: {better_train_acc * 100:.2f}%, Test Accuracy: {better_test_acc * 100:.2f}%")

    # Plot 
    plt.figure(figsize=(6, 6))
    plt.scatter(X_train.numpy()[:, 0][y_train.squeeze() == 0], 
                X_train.numpy()[:, 1][y_train.squeeze() == 0], 
                color='red', label='Class 0 (Train)')
    plt.scatter(X_train.numpy()[:, 0][y_train.squeeze() == 1], 
                X_train.numpy()[:, 1][y_train.squeeze() == 1], 
                color='blue', label='Class 1 (Train)')
    
    # Draw decision boundary
    circle = plt.Circle((0, 0), 0.7, color='yellow', fill=False, label='Decision Boundary')
    plt.gca().add_artist(circle)
    plt.title("Better Classifier - Training Data with Decision Boundary")
    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Plot test scatter with decision boundary
    plt.figure(figsize=(6, 6))
    plt.scatter(X_test.numpy()[:, 0][y_test.squeeze() == 0], 
                X_test.numpy()[:, 1][y_test.squeeze() == 0], 
                color='orange', label='Class 0 (Test)')
    plt.scatter(X_test.numpy()[:, 0][y_test.squeeze() == 1], 
                X_test.numpy()[:, 1][y_test.squeeze() == 1], 
                color='green', label='Class 1 (Test)')
   
    # Draw decision boundary
    circle = plt.Circle((0, 0), 0.7, color='yellow', fill=False, label='Decision Boundary')
    plt.gca().add_artist(circle)
    plt.title("Better Classifier - Test Data with Decision Boundary")
    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
