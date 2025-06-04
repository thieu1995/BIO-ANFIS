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
from mealpy import get_optimizer_by_class, Optimizer, get_all_optimizers, FloatVar
from permetrics import ClassificationMetric, RegressionMetric
from xanfis.helpers import membership_family as mfam
from xanfis.helpers.metric_util import get_all_regression_metrics, get_all_classification_metrics
from xanfis.helpers import validator


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
    A customizable Adaptive Neuro-Fuzzy Inference System (ANFIS) model implemented in PyTorch.

    This class implements a modular and flexible ANFIS architecture that supports different membership functions,
    activation functions, task types (classification, binary classification, regression), and rule strengths
    calculation strategies (to address vanishing gradient issues in fuzzy logic models).

    Parameters:
        input_dim (int): Number of input features.
        num_rules (int): Number of fuzzy rules in the ANFIS model.
        output_dim (int): Number of output units (e.g., number of classes for classification tasks).
        mf_class (str or mfam.BaseMembership, optional): The membership function class to use. Can be a string
            referring to a predefined membership function name or an instance of a custom membership class.
        task (str, optional): Type of learning task: 'classification', 'binary_classification', or 'regression'.
            Default is 'classification'.
        act_output (str, optional): Activation function to apply at the output layer. If not provided, a default
            activation is chosen based on the task (Softmax for classification, Sigmoid for binary classification,
            Identity for regression).
        vanishing_strategy (str, optional): Strategy for computing rule strength to mitigate vanishing gradient
            issues. Supported values: 'prod', 'mean', 'blend'. Default is 'prod'.
        reg_lambda (float, optional): Regularization strength for L2-regularized least squares when updating
            consequent parameters. Default is 0 (no regularization).
        seed (int, optional): Random seed for reproducibility.
        **kwargs: Additional arguments reserved for future compatibility.

    Attributes:
        memberships (nn.ModuleList): List of membership function modules for each fuzzy rule.
        coeffs (nn.Parameter): Learnable parameters (consequents) representing the linear coefficients per rule.
        act_output_ (nn.Module): Activation function module used in the output layer.
        mf_class_ (type): Class object of the selected membership function.
        _get_strength (Callable): Method used for computing rule strength (based on chosen strategy).

    Supported Membership Functions:
        - Gaussian
        - Trapezoidal
        - Triangular
        - Sigmoid
        - Bell
        - GBell
        - PiShaped
        - SShaped
        - ZShaped
        - Linear

    Supported Output Activations:
        Any activation function in torch.nn.modules.activation, including:
        ReLU, Sigmoid, Tanh, GELU, Softmax, Identity, etc.

    Supported Vanishing Strategies:
        - 'prod': Product of membership values (classical approach).
        - 'mean': Mean of membership values.
        - 'blend': A learned blend between product and mean based on log-strength scaling.

    Example:
        >>> model = CustomANFIS(input_dim=4, num_rules=5, output_dim=3, mf_class="Gaussian",
        >>>                     task="classification", act_output="Softmax", vanishing_strategy="blend")
        >>> output = model(torch.randn(32, 4))
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
                 task="classification", vanishing_strategy="prod", act_output=None, reg_lambda=None,
                 seed=None, **kwargs):
        """
        Initialize a customizable multi-layer perceptron (ANFIS) model.
        """
        super(CustomANFIS, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rules = num_rules
        self.task = task
        self.act_output = act_output
        self.seed = seed
        if reg_lambda is None:
            self.reg_lambda = 0.
        else:
            self.reg_lambda = reg_lambda
        self.kwargs = kwargs

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if vanishing_strategy is None:
            self.vanishing_strategy = "prod"

        if validator.is_str_in_sequence(vanishing_strategy, self.SUPPORTED_VANISHING_STRATEGIES):
            self.vanishing_strategy = vanishing_strategy
            if self.vanishing_strategy == "prod":
                self._get_strength = self.__get_strength_by_prod
            elif self.vanishing_strategy == "mean":
                self._get_strength = self.__get_strength_by_mean
            else:
                self._get_strength = self.__get_strength_by_blend
        else:
            raise ValueError(f"Unsupported vanishing strategy: {vanishing_strategy}. Supported strategies are: {self.SUPPORTED_VANISHING_STRATEGIES}")

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
            self.act_output_ = self._get_act(act_output)

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

    def __get_strength_by_mean(self, memberships):
        """
        Calculate the strengths of the rules using mean method.

        Parameters:
            - memberships (torch.Tensor): Membership values for each rule.

        Returns:
            - torch.Tensor: Strengths of the rules.
        """
        # Calculate strengths by taking the mean along the input dimension (dim=2)
        return torch.mean(memberships, dim=2)

    def __get_strength_by_prod(self, memberships):
        """
        Calculate the strengths of the rules using product method.

        Parameters:
            - memberships (torch.Tensor): Membership values for each rule.

        Returns:
            - torch.Tensor: Strengths of the rules.
        """
        # Calculate rule strengths by taking the product along the input dimension (dim=2)
        return torch.prod(memberships, dim=2)

    def __get_strength_by_blend(self, memberships):
        """
        Calculate the strengths of the rules using blend method.

        Parameters:
            - memberships (torch.Tensor): Membership values for each rule.

        Returns:
            - torch.Tensor: Strengths of the rules.
        """
        # Calculate strengths using a blend of product and mean
        prod_strengths = torch.prod(memberships, dim=2)
        mean_strengths = torch.mean(memberships, dim=2)
        # Compute blending factor alpha based on log of product strength
        log_strength = torch.log(prod_strengths + 1e-8)
        alpha = torch.sigmoid(-10 * (log_strength + 6))
        return (1 - alpha) * prod_strengths + alpha * mean_strengths

    def _get_membership_strengths(self, X):
        # Layer 1: Calculate membership values for all rules (N x num_rules x input_dim)
        memberships = torch.stack([membership(X) for membership in self.memberships], dim=1)

        # Layer 2: Calculate rule strengths
        strengths = self._get_strength(memberships)

        # Layer 3: Normalize strengths (N x num_rules)
        strengths_sum = torch.sum(strengths, dim=1, keepdim=True)
        return strengths / (strengths_sum + 1e-8)       # normalized_strengths

    def forward(self, X):
        """
        Forward pass through the Anfis model.

        Parameters:
            - x (torch.Tensor): The input tensor.

        Returns:
            - torch.Tensor: The output of the ANFIS model.
        """
        # Layer 1: Calculate membership values for all rules (N x num_rules x input_dim)
        # Layer 2: Calculate rule strengths
        # Layer 3: Normalize strengths (N x num_rules)
        normalized_strengths = self._get_membership_strengths(X)

        # Layer 4: Prepare input for consequent layer  ==> (N, num_rules, input_dim)
        weighted_inputs = torch.einsum("ni,rij->nrj", X, self.coeffs[:, :-1, :]) + self.coeffs[:, -1, :]  # (N, num_rules, output_dim)

        # Layer 5: Apply normalized weights to rule outputs
        # Multiply with normalized strengths (broadcasting): (N, num_rules, 1) * (N, num_rules, output_dim)
        output = torch.sum(normalized_strengths.unsqueeze(-1) * weighted_inputs, dim=1)  # (N, output_dim)

        # Layer 6: Apply activation function for the output layer
        output = self.act_output_(output)  # (N, output_dim)
        return output

    def update_output_weights_by_least_squares(self, X, y):
        with torch.no_grad():
            N = X.shape[0]
            ones = torch.ones(N, 1)
            X_bias = torch.cat([X, ones], dim=1)        # (N, input_dim + 1)
            normalized_strengths = self._get_membership_strengths(X)        # (N x num_rules)

            # Vectorized: Multiply strengths with X_bias
            X_expanded = X_bias.unsqueeze(1)  # (N, 1, input_dim + 1)
            strengths_expanded = normalized_strengths.unsqueeze(2)  # (N, num_rules, 1)
            F = strengths_expanded * X_expanded  # (N, num_rules, input_dim + 1)
            F = F.reshape(N, -1)  # (N, num_rules * (input_dim + 1))
            if self.task =="classification" and (y.ndim == 1 or y.shape[1] == 1):
                y = torch.nn.functional.one_hot(y.squeeze().long(), num_classes=self.output_dim).float()
            else:
                y = y.to(dtype=F.dtype)

            if self.reg_lambda == 0:    # No regularization
                # coeffs_flat = torch.linalg.lstsq(F, y, driver='gelsd').solution
                coeffs_flat = torch.linalg.pinv(F) @ y

            else:   # Least Squares Estimation with L2
                I = torch.eye(F.shape[1], device=F.device, dtype=F.dtype)
                coeffs_flat = torch.linalg.solve(F.T @ F + self.reg_lambda * I, F.T @ y)

            coeffs = coeffs_flat.view(self.num_rules, self.input_dim + 1, self.output_dim)
            self.coeffs.data.copy_(coeffs)

    def set_weights(self, solution):
        """
        Set only the premise (non-consequent) weights of the network based on a given solution vector.

        Parameters:
            - solution (np.ndarray): A flat array of weights to set in the model (excluding consequent weights).
        """
        with torch.no_grad():
            idx = 0
            for name, param in self.named_parameters():
                if 'coeffs' in name:
                    continue  # Skip consequent parameters
                param_size = param.numel()
                param.copy_(torch.tensor(solution[idx:idx + param_size], dtype=param.dtype, device=param.device).view(param.shape))
                idx += param_size

    def get_weights(self):
        """
        Retrieve only the premise (non-consequent) weights as a flattened NumPy array.

        Returns:
            - np.ndarray: Flattened array of the model's premise weights.
        """
        weights = []
        for name, param in self.named_parameters():
            if 'coeffs' in name:
                continue  # Skip consequent parameters
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def get_weights_size(self):
        """
        Calculate the number of trainable premise (non-consequent) parameters in the model.

        Returns:
            - int: Total number of premise parameters.
        """
        return sum(
            param.numel() for name, param in self.named_parameters()
            if param.requires_grad and 'coeffs' not in name
        )

class BaseAnfis(BaseEstimator):
    """
    BaseAnfis is a scikit-learn style base class for managing and training
    an Adaptive Neuro-Fuzzy Inference System (ANFIS) model.

    This class provides a high-level interface for building and evaluating ANFIS models using PyTorch
    while retaining compatibility with scikit-learn-style APIs (e.g., `fit`, `predict`, `score`).
    It includes functionality for saving/loading models, metrics evaluation, and logging training results.

    Parameters
    ----------
    num_rules : int
        Number of fuzzy rules in the ANFIS model.
    mf_class : str or object
        The membership function class to use. Can be a string name of a predefined MF type or a custom MF instance.
    task : str, optional (default="classification")
        Type of supervised learning task. One of {"classification", "binary_classification", "regression"}.
    vanishing_strategy : str or None, optional
        Strategy to handle vanishing gradients when combining membership values.
        Can be one of {"prod", "mean", "blend"}. default='prod')
    act_output : str or None, optional
        Output activation function. If None, will be inferred based on task type.
    reg_lambda : float or None, optional (default=0.0)
        Regularization term used when solving for consequent parameters via least squares.
    seed : int or None, optional
        Random seed for reproducibility.

    Attributes
    ----------
    network : torch.nn.Module or None
        The underlying ANFIS model instance. Must be assigned by the subclass or during training.
    loss_train : list or None
        A list to store training loss per epoch (if tracking is implemented).
    SUPPORTED_CLS_METRICS : list of str
        List of supported classification evaluation metrics.
    SUPPORTED_REG_METRICS : list of str
        List of supported regression evaluation metrics.

    Notes
    -----
    - This class is designed to be inherited and extended. The core methods `fit`, `predict`, and `score` must be
      implemented in a subclass or concrete version.
    - Helper methods for saving/loading models and logging training history are included.
    - The `evaluate` method provides metric evaluation for classification and regression tasks.
    """


    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()

    def __init__(self, num_rules, mf_class, task="classification", vanishing_strategy="prod",
                 act_output=None, reg_lambda=None, seed=None):
        self.num_rules = num_rules
        self.mf_class = mf_class
        self.task = task
        self.act_output = act_output
        self.vanishing_strategy = vanishing_strategy
        if reg_lambda is None:
            self.reg_lambda = 0.
        else:
            self.reg_lambda = reg_lambda
        self.seed = seed
        self.network = None
        self.loss_train = None

    @staticmethod
    def _check_method(method=None, list_supported_methods=None):
        """
        Validates if the given method is supported.

        Parameters
        ----------
        method : str
            The method to be checked.
        list_supported_methods : list of str
            A list of supported method names.

        Returns
        -------
        bool
            True if the method is supported; otherwise, raises ValueError.
        """
        if type(method) is str:
            return validator.check_str("method", method, list_supported_methods)
        else:
            raise ValueError(f"method should be a string and belong to {list_supported_methods}")

    def set_seed(self, seed):
        """
        Set the random seed for the model to ensure reproducibility.

        Parameters:
            seed (int, None): The seed value to use for random number generators within the model.

        Notes:
            - This method stores the seed value in the `self.seed` attribute.
            - Setting a seed helps achieve reproducible results, especially in
              training neural networks where randomness affects initialization and
              other stochastic operations.
        """
        self.seed = seed

    def fit(self, X, y):
        """
        Train the ANFIS model on the given dataset.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Training features.
        y : array-like or torch.Tensor
            Target values.
        """
        pass

    def predict(self, X):
        """
        Generate predictions for input data using the trained model.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input features for prediction.

        Returns
        -------
        array-like or torch.Tensor
            Model predictions for each input sample.
        """
        pass

    def score(self, X, y):
        """
        Evaluate the model on the given dataset.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Evaluation features.
        y : array-like or torch.Tensor
            True values.

        Returns
        -------
        float
            The accuracy or evaluation score.
        """
        pass

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """
        Evaluate regression performance metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted values.
        list_metrics : tuple of str, list of str
            List of metrics for evaluation (e.g., "MSE" and "MAE").

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Evaluate classification performance metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted labels.
        list_metrics : tuple of str, list of str
            List of metrics for evaluation (e.g., "AS" and "RS").

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """
        Evaluate the model using specified metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Model's predicted values.
        list_metrics : list of str, optional
            Names of metrics for evaluation (e.g., "MSE", "MAE").

        Returns
        -------
        dict
            Evaluation metrics and their values.
        """
        pass

    def save_training_loss(self, save_path="history", filename="loss.csv"):
        """
        Save training loss history to a CSV file.

        Parameters
        ----------
        save_path : str, optional
            Path to save the file (default: "history").
        filename : str, optional
            Filename for saving loss history (default: "loss.csv").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} model doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_evaluation_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to a CSV file.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.
        y_pred : array-like
            Model predictions.
        list_metrics : list of str, optional
            Metrics for evaluation (default: ("RMSE", "MAE")).
        save_path : str, optional
            Path to save the file (default: "history").
        filename : str, optional
            Filename for saving metrics (default: "metrics.csv").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save true and predicted values to a CSV file.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input features.
        y_true : array-like
            True values.
        save_path : str, optional
            Path to save the file (default: "history").
        filename : str, optional
            Filename for saving predicted values (default: "y_predicted.csv").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="model.pkl"):
        """
        Save the trained model to a pickle file.

        Parameters
        ----------
        save_path : str, optional
            Path to save the model (default: "history").
        filename : str, optional
            Filename for saving model, with ".pkl" extension (default: "model.pkl").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="model.pkl") -> EstimatorType:
        """
        Load a model from a pickle file.

        Parameters
        ----------
        load_path : str, optional
            Path to load the model from (default: "history").
        filename : str, optional
            Filename of the saved model (default: "model.pkl").

        Returns
        -------
        BaseAnfis
            The loaded model.
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))


class BaseClassicAnfis(BaseAnfis):
    """
    A classical ANFIS (Adaptive Neuro-Fuzzy Inference System) model for classification tasks
    with hybrid learning: gradient descent for premise parameters and least squares estimation
    for consequent parameters.

    This implementation supports customizable optimizers, early stopping, L2 regularization, and validation split.

    Parameters
    ----------
    num_rules : int, default=10
        Number of fuzzy rules in the ANFIS model.
    mf_class : str, default="Gaussian"
        Type of membership function to use (e.g., "Gaussian", "Triangular").
    vanishing_strategy : str or None, optional
        Strategy to handle vanishing gradients when combining membership values.
        Can be one of {"prod", "mean", "blend"}. default='prod')
    act_output : callable or None, default=None
        Activation function to apply to the output layer (e.g., softmax for classification).
    reg_lambda : float or None, optional
        L2 regularization strength. If None, regularization is disabled.
    epochs : int, default=1000
        Number of training epochs.
    batch_size : int, default=16
        Batch size used for training.
    optim : str, default="Adam"
        Name of the optimizer. Must be one of the supported optimizers.
    optim_params : dict or None, default=None
        Dictionary of optimizer hyperparameters (e.g., learning rate).
    early_stopping : bool, default=True
        Whether to apply early stopping during training.
    n_patience : int, default=10
        Number of epochs to wait before early stopping if no improvement.
    epsilon : float, default=0.001
        Minimum change in loss to qualify as improvement for early stopping.
    valid_rate : float, default=0.1
        Percentage of data to use for validation split.
    seed : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress during training.

    Attributes
    ----------
    network : CustomANFIS
        The core ANFIS model with fuzzy rules and trainable layers.
    optimizer : torch.optim.Optimizer
        Optimizer instance based on the specified strategy.
    criterion : torch.nn.Module
        Loss function used (e.g., CrossEntropyLoss for classification).
    early_stopper : EarlyStopper or None
        Instance of early stopping monitor (if enabled).
    """

    SUPPORTED_OPTIMIZERS = [
        "Adafactor", "Adadelta", "Adagrad", "Adam",
        "Adamax", "AdamW", "ASGD", "LBFGS", "NAdam",
        "RAdam", "RMSprop", "Rprop", "SGD", "SparseAdam",
    ]

    def __init__(self, num_rules=10, mf_class="Gaussian", vanishing_strategy="prod", act_output=None,
                 reg_lambda=None, epochs=1000, batch_size=16, optim="Adam", optim_params=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        """
        Initialize the ANFIS with user-defined architecture, training parameters, and optimization settings.
        """
        super().__init__(num_rules, mf_class, "classification", vanishing_strategy=vanishing_strategy,
                         act_output=act_output, reg_lambda=reg_lambda, seed=seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.optim_params = optim_params if optim_params else {}
        self.early_stopping = early_stopping
        self.n_patience = n_patience
        self.epsilon = epsilon
        self.valid_rate = valid_rate
        self.verbose = verbose

        # Internal attributes for model, optimizer, and early stopping
        self.size_input = None
        self.size_output = None
        self.network = None
        self.optimizer = None
        self.criterion = None
        self.patience_count = None
        self.valid_mode = False
        self.early_stopper = None

    def build_model(self):
        """
        Construct the ANFIS model, optimizer, and loss criterion.

        This method:
        - Instantiates the ANFIS network based on current configuration.
        - Configures the optimizer for trainable (non-consequent) parameters.
        - Selects the loss function based on task type.
        - Initializes early stopping if enabled.
        """
        if self.early_stopping:
            # Initialize early stopper if early stopping is enabled
            self.early_stopper = EarlyStopper(patience=self.n_patience, epsilon=self.epsilon)

        # Define model, optimizer, and loss criterion based on task
        self.network = CustomANFIS(self.size_input, self.num_rules, self.size_output, self.mf_class,
                                   self.task, self.vanishing_strategy, self.act_output, self.reg_lambda, self.seed)
        # Freeze consequent parameters during GD
        params = [p for name, p in self.network.named_parameters() if 'coeffs' not in name]
        self.optimizer = getattr(torch.optim, self.optim)(params, **self.optim_params)

        # Select loss function based on task type
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == "binary_classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

    def process_data(self, X, y, **kwargs):
        """
        Process and prepare data for training.

        Parameters
        ----------
        X : array-like
            Feature data for training.
        y : array-like
            Target labels or values for training.
        **kwargs : additional keyword arguments
            Additional parameters for data processing, if needed.
        """
        pass  # Placeholder for data processing logic

    def _fit(self, data, **kwargs):
        """
        Train the ANFIS model using hybrid learning: gradient descent for the
        premise parameters and least squares estimation for consequent parameters.

        Parameters
        ----------
        data : tuple
            A tuple containing (train_loader, X_valid_tensor, y_valid_tensor) for training and validation.
        **kwargs : additional keyword arguments
            Additional parameters for training, if needed.

        Notes
        -----
        - Early stopping is applied if enabled.
        - Training loss (and validation loss, if applicable) is printed per epoch when verbose=True.
        - Least squares estimation is applied at each batch step to update the consequent parameters.
        """
        # Unpack training and validation data
        train_loader, X_valid_tensor, y_valid_tensor = data

        # Start training
        self.network.train()  # Set model to training mode
        for epoch in range(self.epochs):
            # Initialize total loss for this epoch
            total_loss = 0.0

            # # Update consequent parameters using LSE for all training data
            # X_all = torch.cat([x for x, _ in train_loader], dim=0)
            # y_all = torch.cat([y for _, y in train_loader], dim=0)
            # self.network.update_output_weights_by_least_squares(X_all.detach(), y_all.detach())

            # Training step over batches
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()  # Clear gradients
                # Update consequent parameters using LSE
                self.network.update_output_weights_by_least_squares(batch_X.detach(), batch_y.detach())

                # Forward pass
                pred = self.network(batch_X)

                # Compute loss with L2 regularization (only for trainable parameters)
                l2_reg = sum(torch.sum(param ** 2) for name, param in self.network.named_parameters() if
                             param.requires_grad and 'coeffs' not in name)
                loss = self.criterion(pred, batch_y)  + self.reg_lambda * l2_reg # Compute loss

                # Backpropagation and optimization
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()  # Accumulate batch loss

            # Calculate average training loss for this epoch
            avg_loss = total_loss / len(train_loader)

            # Perform validation if validation mode is enabled
            if self.valid_mode:
                self.network.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    val_output = self.network(X_valid_tensor)
                    val_loss = self.criterion(val_output, y_valid_tensor)

                # Early stopping based on validation loss
                if self.early_stopping and self.early_stopper.early_stop(val_loss):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                if self.verbose:
                    print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
            else:
                # Early stopping based on training loss if no validation is used
                if self.early_stopping and self.early_stopper.early_stop(avg_loss):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                if self.verbose:
                    print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}")

            # Return to training mode for next epoch
            self.network.train()


class BaseGdAnfis(BaseAnfis):
    """
    A gradient-based Adaptive Neuro-Fuzzy Inference System (ANFIS) base class using PyTorch.

    This class supports training an ANFIS model using various gradient descent optimizers
    provided by PyTorch. It includes options for early stopping, regularization, and model
    validation.

    Attributes
    ----------
    SUPPORTED_OPTIMIZERS : list of str
        List of supported PyTorch optimizer names.

    epochs : int
        Number of training epochs.

    batch_size : int
        Mini-batch size used for gradient-based optimization.

    optim : str
        Name of the optimizer to use (must be in SUPPORTED_OPTIMIZERS).

    optim_params : dict
        Parameters to initialize the optimizer.

    early_stopping : bool
        Whether to apply early stopping during training.

    n_patience : int
        Number of epochs with no improvement after which training is stopped.

    epsilon : float
        Minimum change in the monitored loss to qualify as an improvement.

    valid_rate : float
        Proportion of the training data to be used for validation.

    verbose : bool
        Whether to print logs during training.

    size_input : int
        Number of input features (set during data processing).

    size_output : int
        Number of output features (set during data processing).

    network : CustomANFIS
        The ANFIS model instance.

    optimizer : torch.optim.Optimizer
        The PyTorch optimizer instance.

    criterion : torch.nn.Module
        Loss function used for training.

    patience_count : int or None
        Counter for tracking early stopping patience.

    valid_mode : bool
        Whether validation mode is enabled.

    early_stopper : EarlyStopper or None
        Instance managing early stopping logic.

    Parameters
    ----------
    num_rules : int, optional
        Number of fuzzy rules (default is 10).

    mf_class : str, optional
        Type of membership function to use (default is "Gaussian").

    vanishing_strategy : str, optional
        Strategy to compute rule strengths (to avoid gradient vanishing too), e.g., "prod" or "min" (default is "prod").

    act_output : callable or None, optional
        Activation function for the output layer (default is None).

    reg_lambda : float or None, optional
        Regularization parameter for L2 loss (default is None).

    epochs : int, optional
        Number of training epochs (default is 1000).

    batch_size : int, optional
        Batch size for training (default is 16).

    optim : str, optional
        Optimizer name from SUPPORTED_OPTIMIZERS (default is "Adam").

    optim_params : dict or None, optional
        Parameters for optimizer initialization (default is None).

    early_stopping : bool, optional
        Enable or disable early stopping (default is True).

    n_patience : int, optional
        Patience threshold for early stopping (default is 10).

    epsilon : float, optional
        Minimum improvement threshold for early stopping (default is 0.001).

    valid_rate : float, optional
        Validation split ratio from training data (default is 0.1).

    seed : int, optional
        Random seed for reproducibility (default is 42).

    verbose : bool, optional
        Enable verbose output (default is True).

    Methods
    -------
    build_model():
        Build and initialize the ANFIS network, optimizer, and loss function.

    process_data(X, y, **kwargs):
        Prepares input features and targets for training (to be implemented in subclass).

    _fit(data, **kwargs):
        Trains the ANFIS model using mini-batch gradient descent and early stopping.
    """

    SUPPORTED_OPTIMIZERS = [
        "Adafactor", "Adadelta", "Adagrad", "Adam",
        "Adamax", "AdamW", "ASGD", "LBFGS", "NAdam",
        "RAdam", "RMSprop", "Rprop", "SGD", "SparseAdam",
    ]

    def __init__(self, num_rules=10, mf_class="Gaussian", vanishing_strategy="prod", act_output=None,
                 reg_lambda=None, epochs=1000, batch_size=16, optim="Adam", optim_params=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        """
        Initialize the ANFIS with user-defined architecture, training parameters, and optimization settings.
        """
        super().__init__(num_rules, mf_class, "classification", vanishing_strategy=vanishing_strategy,
                         act_output=act_output, reg_lambda=reg_lambda, seed=seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.optim_params = optim_params if optim_params else {}
        self.early_stopping = early_stopping
        self.n_patience = n_patience
        self.epsilon = epsilon
        self.valid_rate = valid_rate
        self.verbose = verbose

        # Internal attributes for model, optimizer, and early stopping
        self.size_input = None
        self.size_output = None
        self.network = None
        self.optimizer = None
        self.criterion = None
        self.patience_count = None
        self.valid_mode = False
        self.early_stopper = None

    def build_model(self):
        """
        Build and initialize the ANFIS model, optimizer, and criterion based on user specifications.

        This function sets up the model structure, optimizer type and parameters,
        and loss criterion depending on the task type (classification or regression).
        """
        if self.early_stopping:
            # Initialize early stopper if early stopping is enabled
            self.early_stopper = EarlyStopper(patience=self.n_patience, epsilon=self.epsilon)

        # Define model, optimizer, and loss criterion based on task
        self.network = CustomANFIS(self.size_input, self.num_rules, self.size_output, self.mf_class,
                                   self.task, self.vanishing_strategy, self.act_output, self.reg_lambda, self.seed)
        self.optimizer = getattr(torch.optim, self.optim)(self.network.parameters(), **self.optim_params)

        # Select loss function based on task type
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == "binary_classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

    def process_data(self, X, y, **kwargs):
        """
        Process and prepare data for training.

        Parameters
        ----------
        X : array-like
            Feature data for training.
        y : array-like
            Target labels or values for training.
        **kwargs : additional keyword arguments
            Additional parameters for data processing, if needed.
        """
        pass  # Placeholder for data processing logic

    def _fit(self, data, **kwargs):
        """
        Train the ANFIS model on the provided data.

        Parameters
        ----------
        data : tuple
            A tuple containing (train_loader, X_valid_tensor, y_valid_tensor) for training and validation.
        **kwargs : additional keyword arguments
            Additional parameters for training, if needed.
        """
        # Unpack training and validation data
        train_loader, X_valid_tensor, y_valid_tensor = data

        # Start training
        self.network.train()  # Set model to training mode
        for epoch in range(self.epochs):
            # Initialize total loss for this epoch
            total_loss = 0.0

            # Training step over batches
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()  # Clear gradients

                # Step 1: forward pass
                pred = self.network(batch_X)

                # Step 2: compute loss with L2 regularization (only for trainable parameters)
                l2_reg = sum(torch.sum(param ** 2) for name, param in self.network.named_parameters())
                loss = self.criterion(pred, batch_y) + self.reg_lambda * l2_reg  # Compute loss

                # Step 3: Backpropagation and optimization
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()  # Accumulate batch loss

            # Calculate average training loss for this epoch
            avg_loss = total_loss / len(train_loader)

            # Perform validation if validation mode is enabled
            if self.valid_mode:
                self.network.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    val_output = self.network(X_valid_tensor)
                    val_loss = self.criterion(val_output, y_valid_tensor)

                # Early stopping based on validation loss
                if self.early_stopping and self.early_stopper.early_stop(val_loss):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                if self.verbose:
                    print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
            else:
                # Early stopping based on training loss if no validation is used
                if self.early_stopping and self.early_stopper.early_stop(avg_loss):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                if self.verbose:
                    print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}")

            # Return to training mode for next epoch
            self.network.train()


class BaseBioAnfis(BaseAnfis):
    """
    Base class for biologically-inspired ANFIS models using metaheuristic optimization.

    This class serves as a base for integrating Adaptive Neuro-Fuzzy Inference Systems (ANFIS)
    with metaheuristic algorithms (e.g., Genetic Algorithm, PSO, etc.) to optimize membership function
    parameters and rule weights. The consequent parameters are learned using Least Squares Estimation (LSE)
    in a hybrid-learning manner.

    Attributes
    ----------
    SUPPORTED_OPTIMIZERS : list of str
        List of supported optimizer names from the Mealpy library.
    SUPPORTED_CLS_OBJECTIVES : dict
        Dictionary of supported classification metrics from the `permetrics` library.
    SUPPORTED_REG_OBJECTIVES : dict
        Dictionary of supported regression metrics from the `permetrics` library.
    optim : str
        Name of the metaheuristic optimizer used.
    optim_params : dict
        Hyperparameters for the selected optimizer.
    verbose : bool
        Whether to print logs during training.
    size_input : int
        Number of input features (set during model build).
    size_output : int
        Number of output neurons (set during model build).
    network : CustomANFIS
        The core ANFIS network module.
    optimizer : Optimizer
        Metaheuristic optimizer instance.
    obj_name : str
        Name of the optimization objective function.
    metric_class : permetrics object
        Metric computation class for evaluating performance.

    Methods
    -------
    set_optim_and_paras(optim, optim_params)
        Sets the optimizer name and parameters.
    build_model()
        Constructs the ANFIS network and optimizer instance.
    get_name()
        Returns the name of the model based on optimizer settings.
    _set_optimizer(optim, optim_params)
        Internal method to initialize the optimizer.
    _set_lb_ub(lb, ub, n_dims)
        Validates and formats lower and upper bounds for optimization.
    objective_function(solution)
        Computes the loss/fitness value for a given solution vector.
    _fit(data, lb, ub, mode, n_workers, termination, save_population, **kwargs)
        Trains the ANFIS model using a metaheuristic-based optimization strategy.

    Notes
    -----
    - This class supports hybrid learning by combining metaheuristics and LSE.
    - Intended to be extended by specific regressors/classifiers in the library.
    - Requires `CustomANFIS`, `Optimizer` from `mealpy`, and metrics from `permetrics`.
    """

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers(verbose=False).keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()

    def __init__(self, num_rules=10, mf_class="Gaussian", vanishing_strategy="prod", act_output=None,
                 reg_lambda=None, optim="BaseGA", optim_params=None, obj_name=None, seed=42, verbose=True):
        """
        Initialize the BaseGdAnfis model with user-defined architecture and training configurations.

        Parameters
        ----------
        num_rules : int, default=10
            Number of fuzzy rules in the ANFIS network.
        mf_class : str, default="Gaussian"
            Type of membership function to use.
        vanishing_strategy : str or None, optional
            Strategy for handling vanishing gradients (if any).
        act_output : callable or None, default=None
            Activation function applied at the output layer.
        reg_lambda : float or None, optional
            L2 regularization strength; set None to disable.
        epochs : int, default=1000
            Total number of training epochs.
        batch_size : int, default=16
            Batch size for training.
        optim : str, default="Adam"
            Name of optimizer from SUPPORTED_OPTIMIZERS.
        optim_params : dict or None, default=None
            Parameters for the selected optimizer (e.g., {'lr': 0.01}).
        early_stopping : bool, default=True
            Whether to use early stopping.
        n_patience : int, default=10
            Patience for early stopping.
        epsilon : float, default=0.001
            Threshold for early stopping improvement.
        valid_rate : float, default=0.1
            Proportion of training data to use for validation.
        seed : int, default=42
            Random seed for reproducibility.
        verbose : bool, default=True
            Whether to print training logs per epoch.
        """
        super().__init__(num_rules, mf_class, "classification", vanishing_strategy=vanishing_strategy,
                         act_output=act_output, reg_lambda=reg_lambda, seed=seed)
        self.optim = optim
        self.optim_params = optim_params
        self.verbose = verbose

        # Initialize model parameters
        self.size_input = None
        self.size_output = None
        self.network = None
        self.optimizer = None
        self.obj_name = obj_name
        self.metric_class = None

    def set_optim_and_paras(self, optim=None, optim_params=None):
        """
        Sets the `optim` and `optim_params` parameters for this class.

        Parameters
        ----------
        optim : str
            The optimizer name to be set.
        optim_params : dict
            Parameters to configure the optimizer.
        """
        self.optim = optim
        self.optim_params = optim_params

    def _set_optimizer(self, optim=None, optim_params=None):
        """
        Validates the real optimizer based on the provided `optim` and `optim_pras`.

        Parameters
        ----------
        optim : str or Optimizer
            The optimizer name or instance to be set.
        optim_params : dict, optional
            Parameters to configure the optimizer.

        Returns
        -------
        Optimizer
            An instance of the selected optimizer.

        Raises
        ------
        TypeError
            If the provided optimizer is neither a string nor an instance of Optimizer.
        """
        if isinstance(optim, str):
            opt_class = get_optimizer_by_class(optim)
            if isinstance(optim_params, dict):
                return opt_class(**optim_params)
            else:
                return opt_class(epoch=300, pop_size=30)
        elif isinstance(optim, Optimizer):
            if isinstance(optim_params, dict):
                if "name" in optim_params:  # Check if key exists and remove it
                    optim.name = optim_params.pop("name")
                optim.set_parameters(optim_params)
            return optim
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def get_name(self):
        """
        Generate a descriptive name for the ANFIS model based on the optimizer.

        Returns:
            str: A string representing the name of the model, including details
            about the optimizer used. If `self.optim` is a string, the name
            will be formatted as "<self.optim_params>-ANFIS". Otherwise, it will
            return "<self.optimizer.name>-ANFIS", assuming `self.optimizer` is an
            object with a `name` attribute.

        Notes:
            - This method relies on the presence of `self.optim`, `self.optim_params`,
              and `self.optimizer.name` attributes within the model instance.
            - It is intended to provide a consistent naming scheme for model instances
              based on the optimizer configuration.
        """
        return f"{self.optimizer.name}-ANFIS-{self.optim_params}"

    def build_model(self):
        """
        Build and initialize the ANFIS model, optimizer, and loss criterion.

        Notes
        -----
        - Initializes `CustomANFIS` with user settings.
        - Instantiates a PyTorch optimizer for trainable parameters.
        - Sets the appropriate loss function based on task type.
        - Prepares early stopping monitor if enabled.
        """
        self.network = CustomANFIS(self.size_input, self.num_rules, self.size_output, self.mf_class, self.task,
                                   self.vanishing_strategy, self.act_output, self.reg_lambda, self.seed)

        self.optimizer = self._set_optimizer(self.optim, self.optim_params)

    def _set_lb_ub(self, lb=None, ub=None, n_dims=None):
        """
        Validates and sets the lower and upper bounds for optimization.

        Parameters
        ----------
        lb : list, tuple, np.ndarray, int, or float, optional
            The lower bounds.
        ub : list, tuple, np.ndarray, int, or float, optional
            The upper bounds.
        n_dims : int
            The number of dimensions.

        Returns
        -------
        tuple
            A tuple containing validated lower and upper bounds.

        Raises
        ------
        ValueError
            If the bounds are not valid.
        """
        if isinstance(lb, (list, tuple, np.ndarray)) and isinstance(ub, (list, tuple, np.ndarray)):
            if len(lb) == len(ub):
                if len(lb) == 1:
                    lb = np.array(lb * n_dims, dtype=float)
                    ub = np.array(ub * n_dims, dtype=float)
                    return lb, ub
                elif len(lb) == n_dims:
                    return lb, ub
                else:
                    raise ValueError(f"Invalid lb and ub. Their length should be equal to 1 or {n_dims}.")
            else:
                raise ValueError(f"Invalid lb and ub. They should have the same length.")
        elif isinstance(lb, (int, float)) and isinstance(ub, (int, float)):
            lb = (float(lb),) * n_dims
            ub = (float(ub),) * n_dims
            return lb, ub
        else:
            raise ValueError(f"Invalid lb and ub. They should be a number of list/tuple/np.ndarray with size equal to {n_dims}")

    def objective_function(self, solution=None):
        """
        Evaluates the fitness function for classification metrics based on the provided solution.

        Parameters
        ----------
        solution : np.ndarray, default=None
            The proposed solution to evaluate.

        Returns
        -------
        result : float
            The fitness value, representing the loss for the current solution.
        """
        X_train, y_train = self.data
        self.network.set_weights(solution)
        self.network.update_output_weights_by_least_squares(X_train, y_train)
        y_pred = self.network(X_train).detach().cpu().numpy()
        y_train = y_train.detach().cpu().numpy()
        loss_train = self.metric_class(y_train, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return np.mean([loss_train])

    def _fit(self, data, lb=(-1.0,), ub=(1.0,), mode='single', n_workers=None,
             termination=None, save_population=False, **kwargs):
        """
        Train the ANFIS model using gradient descent.

        Parameters
        ----------
        data : tuple
            Tuple of (train_loader, X_valid_tensor, y_valid_tensor), where:
            - train_loader : DataLoader
                Iterable over training mini-batches.
            - X_valid_tensor : torch.Tensor or None
                Validation input features.
            - y_valid_tensor : torch.Tensor or None
                Validation targets.
        **kwargs : dict
            Additional keyword arguments for training configuration.

        Notes
        -----
        - Uses standard gradient descent for training all parameters.
        - Applies L2 regularization to trainable weights if enabled.
        - Supports both training-only and validation-based early stopping.
        - Logs loss and early stopping progress when `verbose=True`.
        """
        # Get data
        n_dims = self.network.get_weights_size()
        lb, ub = self._set_lb_ub(lb, ub, n_dims)
        self.data = data

        log_to = "console" if self.verbose else "None"
        if self.obj_name is None:
            raise ValueError("obj_name can't be None")
        else:
            if self.obj_name in self.SUPPORTED_REG_OBJECTIVES.keys():
                minmax = self.SUPPORTED_REG_OBJECTIVES[self.obj_name]
            elif self.obj_name in self.SUPPORTED_CLS_OBJECTIVES.keys():
                minmax = self.SUPPORTED_CLS_OBJECTIVES[self.obj_name]
            else:
                raise ValueError("obj_name is not supported. Please check the library: permetrics to see the supported objective function.")
        problem = {
            "obj_func": self.objective_function,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": minmax,
            "log_to": log_to,
            "save_population": save_population,
        }
        if termination is None:
            self.optimizer.solve(problem, mode=mode, n_workers=n_workers, seed=self.seed)
        else:
            self.optimizer.solve(problem, mode=mode, n_workers=n_workers, termination=termination, seed=self.seed)
        self.network.set_weights(self.optimizer.g_best.solution)
        self.network.update_output_weights_by_least_squares(data[0], data[1])
        self.loss_train = np.array(self.optimizer.history.list_global_best_fit)
        return self
