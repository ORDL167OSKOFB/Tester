

from matplotlib import pyplot as plt
import numpy as np


def plot_partial_dependence(X, feature, model):
  
    # Calculate the values to use for the grid
    grid = np.linspace(X[feature].min(), X[feature].max(), num=50)

    # Create a copy of the dataset and replace the values of the chosen feature with the grid values
    X_grid = X.copy()
    X_grid[feature] = grid

    # Get the predicted probabilities for each row of the copied dataset
    probas = model.predict_proba(X_grid)[:, 1]

    # Calculate the average predicted probability for each value in the grid
    avg_probas = np.array([probas[X_grid[feature] == val].mean() for val in grid])

    # Plot the grid values against the average predicted probabilities
    plt.plot(grid, avg_probas)
    plt.xlabel(feature)
    plt.ylabel('Predicted probability of Walking')
    plt.title(f'Partial Dependence Plot for {feature}')
    plt.show()