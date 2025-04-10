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


class BaseAnfis(BaseEstimator):
    """
    Base class for Adaptive Neuro-Fuzzy Inference System (ANFIS) models.

    This class provides a foundation for implementing ANFIS models with support for
    training, evaluation, and saving/loading models. It includes methods for handling
    metrics, setting random seeds, and managing training loss history.

    Attributes:
        SUPPORTED_CLS_METRICS (list): List of supported classification metrics.
        SUPPORTED_REG_METRICS (list): List of supported regression metrics.
        num_rules (int): Number of fuzzy rules.
        mf_class (str): Membership function class.
        task (str): Task type, either "classification" or "regression".
        act_output (str or None): Activation function for the output layer.
        vanishing_strategy (str or None): Strategy for calculating rule strengths.
        seed (int or None): Random seed for reproducibility.
        network (nn.Module or None): Neural network representing the ANFIS model.
        loss_train (list or None): Training loss history.

    Methods:
        set_seed(seed): Sets the random seed for reproducibility.
        fit(X, y): Trains the ANFIS model on the given dataset.
        predict(X): Generates predictions for input data.
        score(X, y): Evaluates the model on the given dataset.
        evaluate(y_true, y_pred, list_metrics): Evaluates the model using specified metrics.
        save_training_loss(save_path, filename): Saves training loss history to a CSV file.
        save_evaluation_metrics(y_true, y_pred, list_metrics, save_path, filename): Saves evaluation metrics to a CSV file.
        save_y_predicted(X, y_true, save_path, filename): Saves true and predicted values to a CSV file.
        save_model(save_path, filename): Saves the trained model to a pickle file.
        load_model(load_path, filename): Loads a model from a pickle file.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()

    def __init__(self, num_rules, mf_class, task="classification", act_output=None, vanishing_strategy=None, seed=None):
        self.num_rules = num_rules
        self.mf_class = mf_class
        self.task = task
        self.act_output = act_output
        self.vanishing_strategy = vanishing_strategy
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


class BaseStandardAnfis(BaseAnfis):
    """
    A custom standard ANFIS class that extends the BaseAnfis class with
    additional features such as early stopping, validation, and various supported optimizers.

    Attributes
    ----------
    SUPPORTED_OPTIMIZERS : list
        A list of optimizer names supported by the class.
    epochs : int
        Number of training epochs.
    batch_size : int
        Size of each training batch.
    optim : str
        Name of the optimizer to use from SUPPORTED_OPTIMIZERS.
    optim_params : dict
        Additional parameters for the optimizer.
    early_stopping : bool
        Flag to enable early stopping.
    n_patience : int
        Number of epochs to wait before stopping if no improvement.
    epsilon : float
        Minimum change to qualify as improvement.
    valid_rate : float
        Proportion of data to use for validation.
    verbose : bool
        If True, outputs training progress.
    size_input : int or None
        Number of input features (set during training).
    size_output : int or None
        Number of output features (set during training).
    network : nn.Module or None
        The ANFIS model instance.
    optimizer : torch.optim.Optimizer or None
        The optimizer used for training.
    criterion : nn.Module or None
        The loss function used for training.
    early_stopper : EarlyStopper or None
        Instance of EarlyStopper for managing early stopping.

    Parameters
    ----------
    num_rules : int
        Number of fuzzy rules.
    mf_class : str
        Membership function class.
    act_output : str or None
        Activation function for the output layer.
    vanishing_strategy : str or None
        Strategy for calculating rule strengths.
    epochs : int, optional
        Number of training epochs (default is 1000).
    batch_size : int, optional
        Size of each training batch (default is 16).
    optim : str, optional
        Name of the optimizer to use (default is "Adam").
    optim_params : dict, optional
        Additional parameters for the optimizer (default is None).
    early_stopping : bool, optional
        Flag to enable early stopping (default is True).
    n_patience : int, optional
        Number of epochs to wait before stopping if no improvement (default is 10).
    epsilon : float, optional
        Minimum change to qualify as improvement (default is 0.001).
    valid_rate : float, optional
        Proportion of data to use for validation (default is 0.1).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    verbose : bool, optional
        If True, outputs training progress (default is True).

    Methods
    -------
    build_model():
        Build and initialize the ANFIS model, optimizer, and criterion based on user specifications.
    process_data(X, y, **kwargs):
        Process and prepare data for training.
    _fit(data, **kwargs):
        Train the ANFIS model on the provided data.
    """

    SUPPORTED_OPTIMIZERS = [
        "Adafactor", "Adadelta", "Adagrad", "Adam",
        "Adamax", "AdamW", "ASGD", "LBFGS", "NAdam",
        "RAdam", "RMSprop", "Rprop", "SGD", "SparseAdam",
    ]

    def __init__(self, num_rules=10, mf_class="Gaussian", act_output=None, vanishing_strategy=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_params=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        """
        Initialize the ANFIS with user-defined architecture, training parameters, and optimization settings.
        """
        super().__init__(num_rules, mf_class, "classification", act_output=act_output,
                         vanishing_strategy=vanishing_strategy,seed=seed)
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
                                   self.task, self.act_output, self.vanishing_strategy, self.seed)
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

                # Forward pass
                output = self.network(batch_X)
                loss = self.criterion(output, batch_y)  # Compute loss

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

