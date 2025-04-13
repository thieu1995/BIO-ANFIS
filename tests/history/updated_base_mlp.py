"""
This module implements a customizable Multi-Layer Perceptron (MLP) framework for regression and classification.
It supports both gradient-based and metaheuristic-based training methods.

Classes:
    - CustomMLP: Defines the architecture of the MLP network.
    - BaseMlp: Base class inheriting BaseEstimator from Scikit-Learn.
        - BaseStandardMlp: Implements gradient-based training for MLP.
            - MlpRegressor: Gradient-based MLP regressor.
            - MlpClassifier: Gradient-based MLP classifier.
        - BaseMhaMlp: Implements metaheuristic-based training for MLP.
            - MhaMlpRegressor: Metaheuristic-based MLP regressor.
            - MhaMlpClassifier: Metaheuristic-based MLP classifier.

Dependencies:
    - NumPy
    - PyTorch
    - Scikit-Learn
    - Permetrics for classification and regression metrics
    - Metaheuristics from mealpy library
"""

from typing import TypeVar, Type
import numpy as np
import torch
import torch.nn as nn
from permetrics import ClassificationMetric, RegressionMetric
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from banfis.helpers.metric_util import get_all_regression_metrics, get_all_classification_metrics
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers, FloatVar

# Custom Type for specifying the optimizer type
Opt = TypeVar("Opt", bound=Optimizer)

class CustomMLP(nn.Module):
    """
    A customizable Multi-Layer Perceptron (MLP) architecture.
    
    Attributes:
        layers (nn.ModuleList): List of layers defining the MLP structure.
        dropout (float): Dropout rate used between layers.
    """
    def __init__(self, input_size: int, hidden_layers: list, output_size: int, dropout: float = 0.0):
        """
        Initializes the CustomMLP.

        Args:
            input_size (int): Number of input features.
            hidden_layers (list): List specifying the number of neurons in each hidden layer.
            output_size (int): Number of output units.
            dropout (float): Dropout rate to prevent overfitting.
        """
        super(CustomMLP, self).__init__()
        # Define the network layers based on input, hidden, and output specifications
        self.layers = nn.ModuleList()
        in_size = input_size
        for h_size in hidden_layers:
            self.layers.append(nn.Linear(in_size, h_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            in_size = h_size
        self.layers.append(nn.Linear(in_size, output_size))

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through MLP layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

# Additional classes would follow the same pattern, including docstrings, attribute descriptions, 
# and detailed comments explaining each part of their initialization and functionality.
