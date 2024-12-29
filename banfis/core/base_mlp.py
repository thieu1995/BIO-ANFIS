#!/usr/bin/env python
# Created by "Thieu" at 14:17, 26/10/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from typing import TypeVar
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers, FloatVar
from permetrics import ClassificationMetric, RegressionMetric
from banfis.helpers.metric_util import get_all_regression_metrics, get_all_classification_metrics
from banfis.helpers import validator

# Create a TypeVar for the base class
EstimatorType = TypeVar('EstimatorType', bound='BaseMlp')


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


class CustomMLP(nn.Module):
    """
    A customizable multi-layer perceptron (MLP) model with flexible hidden layers, activations,
    and dropout rates, suitable for various tasks such as classification and regression.

    Attributes:
        - SUPPORTED_ACTIVATIONS (list of str): A list of supported activation function names.
        - network (nn.Sequential): The constructed MLP network with layers.

    Parameters:
        - size_input (int): Number of input features.
        - size_output (int): Number of output nodes.
        - hidden_layers (list of int): Number of nodes in each hidden layer.
        - act_names (list of str or nn.Module): Activation functions for each hidden layer.
        - dropout_rates (list of float): Dropout rates for each hidden layer.
        - task (str): Task type, "classification", "binary_classification", or "regression".
        - act_output (str or None): Activation function for the output layer; uses default if None.
        - seed (int or None): The random seed for reproducibility
    """

    SUPPORTED_ACTIVATIONS = [
        "Threshold", "ReLU", "RReLU", "Hardtanh", "ReLU6",
        "Sigmoid", "Hardsigmoid", "Tanh", "SiLU", "Mish", "Hardswish", "ELU",
        "CELU", "SELU", "GLU", "GELU", "Hardshrink", "LeakyReLU",
        "LogSigmoid", "Softplus", "Softshrink", "MultiheadAttention", "PReLU",
        "Softsign", "Tanhshrink", "Softmin", "Softmax", "Softmax2d", "LogSoftmax",
    ]

    def __init__(self, size_input, size_output, hidden_layers, act_names, dropout_rates,
                 task="classification", act_output=None, seed=None):
        """
        Initialize a customizable multi-layer perceptron (MLP) model.
        """
        super(CustomMLP, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Ensure hidden_layers is a valid list, tuple, or numpy array
        if not isinstance(hidden_layers, (list, tuple, np.ndarray, int)):
            raise TypeError('hidden_layers must be a list or tuple or a numpy array.')
        if type(hidden_layers) is int:
            hidden_layers = [hidden_layers]

        # Ensure act_names is a valid list, tuple, numpy array, or str
        if not isinstance(act_names, (list, tuple, np.ndarray, str)):
            raise TypeError('act_names must be a list or tuple or a numpy array or name of activation functions.')
        else:
            if type(act_names) is str:
                act_names = [act_names] * len(hidden_layers)
            elif len(act_names) != len(hidden_layers):
                raise ValueError('if act_names is list, then len(act_names) must equal len(hidden_layers).')

        # Configure dropout rates
        if dropout_rates is None:
            dropout_rates = [0] * len(hidden_layers)
        elif isinstance(dropout_rates, (list, tuple, np.ndarray, float)):
            if type(dropout_rates) is float and 0 < dropout_rates < 1:
                dropout_rates = [dropout_rates] * len(hidden_layers)
            elif len(dropout_rates) != len(hidden_layers):
                raise ValueError('if dropout_rates is list, then len(dropout_rates) must equal len(hidden_layers).')
        else:
            raise TypeError('dropout_rates must be a list or tuple or a numpy array or float.')

        # Determine activation for the output layer based on the task
        if act_output is None:
            if task == 'classification':
                act_out = nn.Softmax(dim=1)
            elif task == 'binary_classification':
                act_out = nn.Sigmoid()
            else:  # regression
                act_out = nn.Identity()
        else:
            act_out = self._get_act(act_output)()

        # Initialize the layers
        layers = []
        in_features = size_input

        # Create each hidden layer with specified activation and dropout
        for out_features, act, dropout in zip(hidden_layers, act_names, dropout_rates):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(self._get_act(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features  # Update input for next layer

        # Add output layer
        layers.append(nn.Linear(in_features, size_output))
        layers.append(act_out)

        # Combine all layers
        self.network = nn.Sequential(*layers)

    def _get_act(self, act_name):
        """
        Retrieve the activation function by name.

        Parameters:
            - act_name (str): Name of the activation function.

        Returns:
            - nn.Module: The activation function module.
        """
        if act_name == "Softmax":
            return nn.Softmax(dim=0)
        elif act_name == "none":
            return nn.Identity()
        else:
            return getattr(nn.modules.activation, act_name)()

    def forward(self, x):
        """
        Forward pass through the MLP model.

        Parameters:
            - x (torch.Tensor): The input tensor.

        Returns:
            - torch.Tensor: The output of the MLP model.
        """
        return self.network(x)

    def set_weights(self, solution):
        """
        Set network weights based on a given solution vector.

        Parameters:
            - solution (np.ndarray): A flat array of weights to set in the model.
        """
        with torch.no_grad():
            idx = 0
            for param in self.network.parameters():
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
        return np.concatenate([param.data.cpu().numpy().flatten() for param in self.network.parameters()])

    def get_weights_size(self):
        """
        Calculate the total number of trainable parameters in the model.

        Returns:
            - int: Total number of parameters.
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


# Custom Sklearn-compatible MLP using PyTorch
class BaseMlp(BaseEstimator):
    """
    A custom MLP model compatible with sklearn, implemented using PyTorch. This class supports
    Multi-Layer Perceptron for both classification and regression tasks, with customizable
    hidden layers, activation functions, and dropout rates.

    Parameters
    ----------
    hidden_layers : list of int
        Specifies the number of nodes in each hidden layer.
    act_names : list of str
        List of activation function names, one for each hidden layer.
    dropout_rates : list of float
        Dropout rates for each hidden layer (0 indicates no dropout).
    task : str, optional
        Task type, either "classification" or "regression". Default is "classification".
    act_output : str or None, optional
        Activation function for the output layer, default depends on the task type.
    seed: int or None, optional
        The seed value for the random number generator.

    Attributes
    ----------
    SUPPORTED_CLS_METRICS : dict
        Supported metrics for classification evaluation.
    SUPPORTED_REG_METRICS : dict
        Supported metrics for regression evaluation.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()

    def __init__(self, hidden_layers, act_names, dropout_rates, task="classification", act_output=None, seed=None):
        self.hidden_layers = hidden_layers
        self.act_names = act_names
        self.dropout_rates = dropout_rates
        self.task = task
        self.act_output = act_output
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
        Train the MLP model on the given dataset.

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

    def save_evaluation_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history",
                                filename="metrics.csv"):
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
        BaseMlp
            The loaded model.
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))


class BaseStandardMlp(BaseMlp):
    """
    A custom standard MLP (Multi-Layer Perceptron) class that extends the BaseMlp class with
    additional features such as early stopping, validation, and various supported optimizers.

    Attributes
    ----------
    SUPPORTED_OPTIMIZERS : list
        A list of optimizer names supported by the class.


    Parameters
    ----------
    hidden_layers : tuple
        Number of neurons in each hidden layer.
    act_names : str
        Activation function(s) for each hidden layer.
    dropout_rates : float
        Dropout rate to prevent overfitting.
    act_output : str
        Activation function for the output layer.
    epochs : int
        Number of training epochs.
    batch_size : int
        Size of each training batch.
    optim : str
        Name of the optimizer to use from SUPPORTED_OPTIMIZERS.
    optim_paras : dict, optional
        Additional parameters for the optimizer.
    early_stopping : bool
        Flag to enable early stopping.
    n_patience : int
        Number of epochs to wait before stopping if no improvement.
    epsilon : float
        Minimum change to qualify as improvement.
    valid_rate : float
        Proportion of data to use for validation.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        If True, outputs training progress.
    """

    SUPPORTED_OPTIMIZERS = [
        "Adafactor", "Adadelta", "Adagrad", "Adam",
        "Adamax", "AdamW", "ASGD", "LBFGS", "NAdam",
        "RAdam", "RMSprop", "Rprop", "SGD", "SparseAdam",
    ]

    def __init__(self, hidden_layers=(100,), act_names="ReLU", dropout_rates=0.2, act_output=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_paras=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        """
        Initialize the MLP with user-defined architecture, training parameters, and optimization settings.
        """
        super().__init__(hidden_layers, act_names, dropout_rates, "classification", act_output, seed=seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.optim_paras = optim_paras if optim_paras else {}
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
        Build and initialize the MLP model, optimizer, and criterion based on user specifications.

        This function sets up the model structure, optimizer type and parameters,
        and loss criterion depending on the task type (classification or regression).
        """
        if self.early_stopping:
            # Initialize early stopper if early stopping is enabled
            self.early_stopper = EarlyStopper(patience=self.n_patience, epsilon=self.epsilon)

        # Define model, optimizer, and loss criterion based on task
        self.network = CustomMLP(self.size_input, self.size_output, self.hidden_layers, self.act_names,
                               self.dropout_rates, self.task, self.act_output, self.seed)
        self.optimizer = getattr(torch.optim, self.optim)(self.network.parameters(), **self.optim_paras)

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
        Train the MLP model on the provided data.

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


class BaseMhaMlp(BaseMlp):
    """
    Base class for Metaheuristic-based MLP models that inherit from BaseMlp.

    Attributes
    ----------
    SUPPORTED_OPTIMIZERS : list
        List of supported optimizer names.
    SUPPORTED_CLS_OBJECTIVES : dict
        Supported objectives for classification tasks.
    SUPPORTED_REG_OBJECTIVES : dict
        Supported objectives for regression tasks.
    SUPPORTED_CLS_METRICS : dict
        Supported metrics for classification evaluation.
    SUPPORTED_REG_METRICS : dict
        Supported metrics for regression evaluation.

    Parameters
    ----------
    hidden_layers : tuple of int
        The number of neurons in each hidden layer.
    act_names : str
        The name of the activation function to be used.
    dropout_rates : float
        The dropout rate for regularization.
    act_output : any, optional
        Activation function for output layer (default is None).
    optim : str
        Name of the optimization algorithm to be used.
    optim_paras : dict, optional
        Parameters for the optimizer (default is None).
    obj_name : str, optional
        Objective name for the model evaluation (default is None).
    seed : int
        Random seed for reproducibility (default is 42).
    verbose : bool
        Whether to print verbose output during training (default is True).

    Methods
    -------
    __init__(hidden_layers, act_names, dropout_rates, act_output, optim, optim_paras, obj_name, seed, verbose):
        Initializes the model parameters and configuration.

    _set_optimizer(optim, optim_paras):
        Sets the optimizer based on the provided name or instance.

    build_model():
        Builds the model architecture and sets the optimizer and loss function.

    _set_lb_ub(lb, ub, n_dims):
        Validates and sets the lower and upper bounds for optimization.

    objective_function(solution):
        Evaluates the fitness function for the given solution.

    _fit(data, lb, ub, mode, n_workers, termination, save_population, **kwargs):
        Fits the model to the provided data using the optimizer.
    """

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers().keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()

    def __init__(self, hidden_layers=(100,), act_names="ELU", dropout_rates=0.2, act_output=None,
                 optim="BaseGA", optim_paras=None, obj_name=None, seed=42, verbose=True):
        """
        Initializes the BaseMhaMlp class.
        """
        super().__init__(hidden_layers, act_names, dropout_rates, "classification", act_output, seed=seed)
        self.optim = optim
        self.optim_paras = optim_paras
        self.verbose = verbose

        # Initialize model parameters
        self.size_input = None
        self.size_output = None
        self.network = None
        self.optimizer = None
        self.obj_name = obj_name
        self.metric_class = None

    def set_optim_and_paras(self, optim=None, optim_paras=None):
        """
        Sets the `optim` and `optim_paras` parameters for this class.

        Parameters
        ----------
        optim : str
            The optimizer name to be set.
        optim_paras : dict
            Parameters to configure the optimizer.
        """
        self.optim = optim
        self.optim_paras = optim_paras

    def _set_optimizer(self, optim=None, optim_paras=None):
        """
        Validates the real optimizer based on the provided `optim` and `optim_pras`.

        Parameters
        ----------
        optim : str or Optimizer
            The optimizer name or instance to be set.
        optim_paras : dict, optional
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
            opt_class = get_optimizer_by_name(optim)
            if isinstance(optim_paras, dict):
                return opt_class(**optim_paras)
            else:
                return opt_class(epoch=300, pop_size=30)
        elif isinstance(optim, Optimizer):
            if isinstance(optim_paras, dict):
                return optim.set_parameters(optim_paras)
            return optim
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def get_name(self):
        """
        Generate a descriptive name for the MLP model based on the optimizer.

        Returns:
            str: A string representing the name of the model, including details
            about the optimizer used. If `self.optim` is a string, the name
            will be formatted as "<self.optim_paras>-MLP". Otherwise, it will
            return "<self.optimizer.name>-MLP", assuming `self.optimizer` is an
            object with a `name` attribute.

        Notes:
            - This method relies on the presence of `self.optim`, `self.optim_paras`,
              and `self.optimizer.name` attributes within the model instance.
            - It is intended to provide a consistent naming scheme for model instances
              based on the optimizer configuration.
        """
        return f"{self.optimizer.name}-MLP-{self.optim_paras}"

    def build_model(self):
        """
        Builds the model architecture and sets the optimizer and loss function based on the task.

        Raises
        ------
        ValueError
            If the task is not recognized.
        """
        self.network = CustomMLP(self.size_input, self.size_output, self.hidden_layers, self.act_names,
                               self.dropout_rates, self.task, self.act_output)

        self.optimizer = self._set_optimizer(self.optim, self.optim_paras)

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
        y_pred = self.network(X_train).detach().cpu().numpy()
        loss_train = self.metric_class(y_train, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return np.mean([loss_train])

    def _fit(self, data, lb=(-1.0,), ub=(1.0,), mode='single', n_workers=None,
             termination=None, save_population=False, **kwargs):
        """
        Fits the model to the provided data using the specified optimizer.

        Parameters
        ----------
        data : tuple
            Training data consisting of features and labels.
        lb : tuple, optional
            Lower bounds for the optimization (default is (-1.0,)).
        ub : tuple, optional
            Upper bounds for the optimization (default is (1.0,)).
        mode : str, optional
            Mode for optimization (default is 'single').
        n_workers : int, optional
            Number of workers for parallel processing (default is None).
        termination : any, optional
            Termination criteria for optimization (default is None).
        save_population : bool, optional
            Whether to save the population during optimization (default is False).
        **kwargs : additional parameters
            Additional parameters for the fitting process.

        Returns
        -------
        self : BaseMhaMlp
            The instance of the fitted model.

        Raises
        ------
        ValueError
            If the objective name is None or not supported.
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
        self.loss_train = np.array(self.optimizer.history.list_global_best_fit)
        return self
