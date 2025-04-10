#!/usr/bin/env python
# Created by "Thieu" at 21:01, 10/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import TypeVar
import inspect
import pprint
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers, FloatVar
from permetrics import ClassificationMetric, RegressionMetric
from banfis.helpers import membership_family as mfam
from banfis.helpers.metric_util import get_all_regression_metrics, get_all_classification_metrics
from banfis.helpers import validator


# Create a TypeVar for the base class
EstimatorType = TypeVar('EstimatorType', bound='BaseAnfis')


class EarlyStopper:
    """
    A utility class for implementing early stopping in training processes to prevent overfitting.

    Attributes:
        - patience (int): Number of consecutive epochs to tolerate no improvement before stopping.
        - epsilon (float): Minimum loss improvement threshold to reset the patience counter.
        - counter (int): Tracks the number of epochs without sufficient improvement.
        - min_loss (float): Keeps track of the minimum observed loss.
    """

    def __init__(self, patience=1, epsilon=0.01):
        """
        Initialize the EarlyStopper with specified patience and epsilon.

        Parameters:
            - patience (int): Maximum number of epochs without improvement before stopping.
            - epsilon (float): Minimum loss reduction to reset the patience counter.
        """
        self.patience = patience
        self.epsilon = epsilon
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss):
        """
        Checks if training should be stopped based on the current loss.

        Parameters:
            - loss (float): The current loss value for the epoch.

        Returns:
            - bool: True if training should stop, False otherwise.
        """
        if loss < self.min_loss:
            # Loss has improved; reset counter and update min_loss
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.epsilon):
            # Loss did not improve sufficiently; increment counter
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class CustomANFIS(nn.Module):
    """
    Custom Adaptive Neuro-Fuzzy Inference System (ANFIS) implementation.

    This class implements a customizable ANFIS model using PyTorch. It supports various membership
    functions, activation functions, and vanishing strategies for rule strength calculation.

    Attributes:
        SUPPORTED_ACTIVATIONS (list): List of supported activation functions.
        SUPPORT_MEMBERSHIP_CLASSES (dict): Dictionary mapping membership function names to their classes.
        SUPPORTED_VANISHING_STRATEGIES (list): List of supported strategies for vanishing rule strengths.

    Parameters:
        input_dim (int): Number of input features.
        num_rules (int): Number of fuzzy rules.
        output_dim (int): Number of output features.
        mf_class (str or BaseMembership): Membership function class or its name.
        task (str): Task type, either "classification", "binary_classification", or "regression".
        act_output (str or None): Activation function for the output layer.
        vanishing_strategy (str): Strategy for calculating rule strengths ("prod", "mean", or "blend").
        seed (int or None): Random seed for reproducibility.

    Methods:
        __repr__(): Returns a string representation of the model.
        _get_act(act_name): Retrieves the activation function by name.
        forward(X): Performs a forward pass through the ANFIS model.
        set_weights(solution): Sets the model's weights from a given solution vector.
        get_weights(): Retrieves the model's weights as a flattened array.
        get_weights_size(): Calculates the total number of trainable parameters in the model.
    """

    SUPPORTED_ACTIVATIONS = [
        "Threshold", "ReLU", "RReLU", "Hardtanh", "ReLU6",
        "Sigmoid", "Hardsigmoid", "Tanh", "SiLU", "Mish", "Hardswish", "ELU",
        "CELU", "SELU", "GLU", "GELU", "Hardshrink", "LeakyReLU",
        "LogSigmoid", "Softplus", "Softshrink", "MultiheadAttention", "PReLU",
        "Softsign", "Tanhshrink", "Softmin", "Softmax", "Softmax2d", "LogSoftmax",
    ]

    SUPPORT_MEMBERSHIP_CLASSES = {
        "Gaussian": "GaussianMembership",
        "Trapezoidal": "TrapezoidalMembership",
        "Triangular": "TriangularMembership",
        "Sigmoid": "SigmoidMembership",
        "Bell": "BellMembership",
        "PiShaped": "PiShapedMembership",
        "SShaped": "SShapedMembership",
        "GBell": "GBellMembership",
        "ZShaped": "ZShapedMembership",
        "Linear": "LinearMembership",
    }

    SUPPORTED_VANISHING_STRATEGIES = ["prod", "mean", "blend"]

    def __init__(self, input_dim=None, num_rules=None, output_dim=None, mf_class=None,
                 task="classification", act_output=None, vanishing_strategy=None, seed=None):
        """
        Initialize a customizable multi-layer perceptron (ANFIS) model.
        """
        super(CustomANFIS, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rules = num_rules
        self.task = task
        self.act_output = act_output
        self.vanishing_strategy = vanishing_strategy
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if self.vanishing_strategy is None:
            self.vanishing_strategy = "prod"

        # Ensure hidden_layers is a valid list, tuple, or numpy array
        if mf_class is None:
            self.mf_class = self.SUPPORT_MEMBERSHIP_CLASSES["Gaussian"]
        elif isinstance(mf_class, str):
            if validator.is_str_in_sequence(mf_class, list(self.SUPPORT_MEMBERSHIP_CLASSES.keys())):
                self.mf_class = self.SUPPORT_MEMBERSHIP_CLASSES[mf_class]
            else:
                raise ValueError(f"Unsupported membership function class: {mf_class}. Supported classes are: {list(self.SUPPORT_MEMBERSHIP_CLASSES.keys())}")
        elif isinstance(mf_class, mfam.BaseMembership):
            self.mf_class = mf_class().name()
        else:
            raise TypeError(f"Unsupported membership function class type: {type(mf_class)}. Expected str or BaseMembership instance.")
        self.mf_class_ = getattr(mfam, self.mf_class)  # Get the class from the string name

        # Determine activation for the output layer based on the task
        if act_output is None:
            if task == 'classification':
                self.act_output_ = nn.Softmax(dim=1)
            elif task == 'binary_classification':
                self.act_output_ = nn.Sigmoid()
            else:  # regression
                self.act_output_ = nn.Identity()
        else:
            self.act_output_ = self._get_act(act_output)()

        # Initialize membership functions
        self.memberships = nn.ModuleList([self.mf_class_(self.input_dim) for _ in range(self.num_rules)])

        # Initialize linear coefficients for each rule
        self.coeffs = nn.Parameter(torch.zeros(num_rules, input_dim + 1, output_dim))  # (num_rules, input_dim+1, output_dim)

    def __repr__(self, **kwargs):
        """Pretty-print parameters like scikit-learn's Estimator.
        """
        param_order = list(inspect.signature(self.__init__).parameters.keys())
        param_dict = {k: getattr(self, k) for k in param_order}

        param_str = ", ".join(f"{k}={repr(v)}" for k, v in param_dict.items())
        if len(param_str) <= 80:
            return f"{self.__class__.__name__}({param_str})"
        else:
            formatted_params = ",\n  ".join(f"{k}={pprint.pformat(v)}" for k, v in param_dict.items())
            return f"{self.__class__.__name__}(\n  {formatted_params}\n)"

    def _get_act(self, act_name):
        """
        Retrieve the activation function by name.

        Parameters:
            - act_name (str): Name of the activation function.

        Returns:
            - nn.Module: The activation function module.
        """
        if act_name == "Softmax":
            return nn.Softmax(dim=1)
        elif act_name == "None":
            return nn.Identity()
        else:
            return getattr(nn.modules.activation, act_name)()

    def forward(self, X):
        """
        Forward pass through the Anfis model.

        Parameters:
            - x (torch.Tensor): The input tensor.

        Returns:
            - torch.Tensor: The output of the ANFIS model.
        """
        # Layer 1: Calculate membership values for all rules (N x num_rules x input_dim)
        memberships = torch.stack([membership(X) for membership in self.memberships], dim=1)

        # Layer 2: Calculate rule strengths
        if self.vanishing_strategy == "prod":
            # Original: Taking the product along the input dimension (dim=2
            strengths = torch.prod(memberships, dim=2)      # Resulting shape: (N, num_rules)
        elif self.vanishing_strategy == "mean":
            # Calculate strengths by taking the mean along the input dimension (dim=2)
            strengths = torch.mean(memberships, dim=2)
        elif self.vanishing_strategy == "blend":
            # Calculate strengths using a blend of product and mean
            prod_strengths = torch.prod(memberships, dim=2)
            mean_strengths = torch.mean(memberships, dim=2)
            # Compute blending factor alpha based on log of product strength
            log_strength = torch.log(prod_strengths + 1e-8)
            alpha = torch.sigmoid(-10 * (log_strength + 6))  # adjust parameters if needed
            strengths = (1 - alpha) * prod_strengths + alpha * mean_strengths
        else:
            raise ValueError(f"Unknown vanishing strategy: {self.vanishing_strategy}")

        # Layer 3: Normalize strengths (N x num_rules)
        strengths_sum = torch.sum(strengths, dim=1, keepdim=True)
        normalized_strengths = strengths / (strengths_sum + 1e-8)

        # Layer 4: Prepare input for consequent layer
        # (N, num_rules, input_dim)
        weighted_inputs = torch.einsum("ni,rij->nrj", X, self.coeffs[:, :-1, :]) + self.coeffs[:, -1,:]  # (N, num_rules, output_dim)

        # Layer 5: Apply normalized weights to rule outputs
        # Multiply with normalized strengths (broadcasting): (N, num_rules, 1) * (N, num_rules, output_dim)
        output = torch.sum(normalized_strengths.unsqueeze(-1) * weighted_inputs, dim=1)  # (N, output_dim)

        # Layer 6: Apply activation function for the output layer
        output = self.act_output_(output)  # (N, output_dim)
        return output

    def set_weights(self, solution):
        """
        Set network weights based on a given solution vector.

        Parameters:
            - solution (np.ndarray): A flat array of weights to set in the model.
        """
        with torch.no_grad():
            idx = 0
            for param in self.parameters():
                param_size = param.numel()
                # Ensure dtype and device consistency
                param.copy_(torch.tensor(solution[idx:idx + param_size], dtype=param.dtype, device=param.device).view(param.shape))
                idx += param_size

    def get_weights(self):
        """
        Retrieve network weights as a flattened array.

        Returns:
            - np.ndarray: Flattened array of the model's weights.
        """
        return np.concatenate([param.data.cpu().numpy().flatten() for param in self.parameters()])

    def get_weights_size(self):
        """
        Calculate the total number of trainable parameters in the model.

        Returns:
            - int: Total number of parameters.
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

