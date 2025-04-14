#!/usr/bin/env python
# Created by "Thieu" at 22:09, 10/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import ClassifierMixin, RegressorMixin
from banfis.models.base_anfis import BaseStandardAnfis


class AnfisClassifier(BaseStandardAnfis, ClassifierMixin):
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS) Classifier that inherits from BaseStandardAnfis and ClassifierMixin.

    This class integrates ANFIS with gradient-based optimization techniques for classification tasks, supporting both
    binary and multi-class classification.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique classes found in the target variable.
    size_input : int
        Number of input features (set during training).
    size_output : int
        Number of output features (set during training).
    task : str
        The type of classification task ("binary_classification" or "classification").
    network : nn.Module
        The ANFIS model instance.

    Parameters
    ----------
    num_rules : int, optional
        Number of fuzzy rules (default is 10).
    mf_class : str, optional
        Membership function class (default is "Gaussian").
    act_output : str, optional
        Activation function for the output layer (default is None).
    vanishing_strategy : str or None, optional
        Strategy for calculating rule strengths (default is None).
    epochs : int, optional
        Number of training epochs (default is 1000).
    batch_size : int, optional
        Batch size used in training (default is 16).
    optim : str, optional
        Optimizer to use, selected from the supported optimizers (default is "Adam").
    optim_params : dict, optional
        Parameters for the optimizer, such as learning rate, beta values, etc. (default is None).
    early_stopping : bool, optional
        If True, training will stop early if validation loss does not improve (default is True).
    n_patience : int, optional
        Number of epochs to wait for an improvement in validation loss before stopping (default is 10).
    epsilon : float, optional
        Minimum improvement in validation loss to continue training (default is 0.001).
    valid_rate : float, optional
        Fraction of data to use for validation (default is 0.1).
    seed : int, optional
        Seed for random number generation (default is 42).
    verbose : bool, optional
        If True, prints training progress and validation loss during training (default is True).

    Methods
    -------
    process_data(X, y, **kwargs):
        Prepares and processes data for training, including optional splitting into validation data.

    fit(X, y, **kwargs):
        Trains the ANFIS model on the provided data.

    predict(X):
        Predicts the class labels for the given input data.

    score(X, y):
        Computes the accuracy score for the classifier.

    predict_proba(X):
        Computes the probability estimates for each class (for classification tasks only).

    evaluate(y_true, y_pred, list_metrics=("AS", "RS")):
        Returns performance metrics for the model on the provided test data.
    """

    def __init__(self, num_rules=10, mf_class="Gaussian", act_output=None, vanishing_strategy=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_params=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        # Call superclass initializer with the specified parameters.
        super().__init__(num_rules, mf_class, act_output, vanishing_strategy,
                         epochs, batch_size, optim, optim_params,
                         early_stopping, n_patience, epsilon, valid_rate, seed, verbose)
        self.classes_ = None

    def process_data(self, X, y, **kwargs):
        """
        Prepares and processes data for training, including optional splitting into validation data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        tuple : (train_loader, X_valid_tensor, y_valid_tensor)
            Data loader for training data, and tensors for validation data (if specified).
        """
        X_valid_tensor, y_valid_tensor, X_valid, y_valid  = None, None, None, None

        # Split data into training and validation sets based on valid_rate
        if self.valid_rate is not None:
            if 0 < self.valid_rate < 1:
                # Activate validation mode if valid_rate is set between 0 and 1
                self.valid_mode = True
                X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.valid_rate,
                                                          random_state=self.seed, shuffle=True, stratify=y)
            else:
                raise ValueError("Validation rate must be between 0 and 1.")

        # Convert data to tensors and set up DataLoader
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        if self.task == "binary_classification":
            y_tensor = torch.tensor(y, dtype=torch.float32)
            y_tensor = torch.unsqueeze(y_tensor, 1)

        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        if self.valid_mode:
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
            if self.task == "binary_classification":
                y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
                y_valid_tensor = torch.unsqueeze(y_valid_tensor, 1)

        return train_loader, X_valid_tensor, y_valid_tensor

    def fit(self, X, y, **kwargs):
        """
        Trains the ANFIS model on the provided data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Set input and output sizes based on data and initialize task
        self.size_input = X.shape[1]
        y = np.squeeze(np.array(y))
        if y.ndim != 1:
            y = np.argmax(y, axis=1)

        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self.task = "binary_classification"
            self.size_output = 1
        else:
            self.task = "classification"
            self.size_output = len(self.classes_)

        # Process data for training and validation
        data = self.process_data(X, y, **kwargs)

        # Build the model architecture
        self.build_model()

        # Train the model using processed data
        self._fit(data, **kwargs)

        return self

    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        numpy.ndarray
            Predicted class labels for each sample.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.network.eval()
        with torch.no_grad():
            output = self.network(X_tensor)  # Get model predictions
            if self.task == "classification":  # Multi-class classification
                _, predicted = torch.max(output, 1)
            else:  # Binary classification
                predicted = (output > 0.5).int().squeeze()
        return predicted.numpy()

    def score(self, X, y):
        """
        Computes the accuracy score for the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            True class labels.

        Returns
        -------
        float
            Accuracy score of the classifier.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict_proba(self, X):
        """
        Computes the probability estimates for each class (for classification tasks only).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        numpy.ndarray
            Probability predictions for each class.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self.task not in ["classification", "binary_classification"]:
            raise ValueError("predict_proba is only available for classification tasks.")

        self.network.eval()  # Set model to evaluation mode
        with torch.no_grad():
            probs = self.network.forward(X_tensor)  # Forward pass to get probability estimates

        return probs.numpy()  # Return as numpy array

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Returns performance metrics for the model on the provided test data.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.

        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        list_metrics : list, default=("AS", "RS")
            List of performance metrics to calculate. Refer to Permetrics (https://github.com/thieu1995/permetrics) library for available metrics.

        Returns
        -------
        dict
            Dictionary with results for the specified metrics.
        """
        return self._BaseAnfis__evaluate_cls(y_true, y_pred, list_metrics)


class AnfisRegressor(BaseStandardAnfis, RegressorMixin):
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS) Regressor for predicting continuous values.

    This class integrates ANFIS with gradient-based optimization techniques for regression tasks, supporting both
    single-output and multi-output regression.

    Attributes
    ----------
    size_input : int
        Number of input features (set during training).
    size_output : int
        Number of output features (set during training).
    task : str
        The type of regression task ("regression" or "multi_regression").
    network : nn.Module
        The ANFIS model instance.

    Parameters
    ----------
    num_rules : int, optional
        Number of fuzzy rules (default is 10).
    mf_class : str, optional
        Membership function class (default is "Gaussian").
    act_output : str, optional
        Activation function for the output layer (default is None).
    vanishing_strategy : str or None, optional
        Strategy for calculating rule strengths (default is None).
    epochs : int, optional
        Number of epochs for training (default is 1000).
    batch_size : int, optional
        Size of the mini-batch during training (default is 16).
    optim : str, optional
        Optimization algorithm (default is "Adam").
    optim_params : dict, optional
        Additional parameters for the optimizer (default is None).
    early_stopping : bool, optional
        Flag to enable early stopping during training (default is True).
    n_patience : int, optional
        Number of epochs to wait for improvement before stopping (default is 10).
    epsilon : float, optional
        Tolerance for improvement (default is 0.001).
    valid_rate : float, optional
        Proportion of data to use for validation (default is 0.1).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    verbose : bool, optional
        Flag to enable verbose output during training (default is True).

    Methods
    -------
    process_data(X, y, **kwargs):
        Prepares the input data for training and validation by converting it to tensors
        and creating DataLoaders for batch processing.

    fit(X, y, **kwargs):
        Fits the ANFIS model to the provided training data.

    predict(X):
        Predicts the target values for the given input features.

    score(X, y):
        Computes the R2 score of the predictions.

    evaluate(y_true, y_pred, list_metrics=("MSE", "MAE")):
        Returns a list of performance metrics for the predictions.
    """

    def __init__(self, num_rules=10, mf_class="Gaussian", act_output=None, vanishing_strategy=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_params=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True):
        super().__init__(num_rules, mf_class, act_output, vanishing_strategy,
                         epochs, batch_size, optim, optim_params,
                         early_stopping, n_patience, epsilon, valid_rate, seed, verbose)

    def process_data(self, X, y, **kwargs):
        """
        Prepares the input data for training and validation by converting it to tensors
        and creating DataLoaders for batch processing.

        Parameters
        ----------
        X : array-like
            Input features for the regression task.
        y : array-like
            Target values for the regression task.

        Returns
        -------
        tuple
            A tuple containing the training DataLoader and optional validation tensors.
        """
        X_valid_tensor, y_valid_tensor, X_valid, y_valid = None, None, None, None

        # Check for validation rate and split the data if applicable
        if self.valid_rate is not None:
            if 0 < self.valid_rate < 1:
                # Split data into training and validation sets
                self.valid_mode = True
                X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.valid_rate,
                                                          random_state=self.seed, shuffle=True)
            else:
                raise ValueError("Validation rate must be between 0 and 1.")

        # Convert training data to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Ensure the target tensor has correct dimensions
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)

        # Create DataLoader for training data
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        # Process validation data if in validation mode
        if self.valid_mode:
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
            if y_valid_tensor.ndim == 1:
                y_valid_tensor = y_valid_tensor.unsqueeze(1)

        return train_loader, X_valid_tensor, y_valid_tensor

    def fit(self, X, y, **kwargs):
        """
        Fits the ANFIS model to the provided training data.

        Parameters
        ----------
        X : array-like
            Input features for training.
        y : array-like
            Target values for training.

        Returns
        -------
        self : AnfisRegressor
            Returns the instance of the fitted model.
        """
        # Check and prepare the input parameters
        self.size_input = X.shape[1]
        y = np.squeeze(np.array(y))
        self.size_output = 1  # Default output size for regression
        self.task = "regression"  # Default task is regression

        if y.ndim == 2:
            # Adjust task if multi-dimensional target is provided
            self.task = "multi_regression"
            self.size_output = y.shape[1]

        # Process data to prepare DataLoader and tensors
        data = self.process_data(X, y, **kwargs)

        # Build the ANFIS model
        self.build_model()

        # Fit the model to the training data
        self._fit(data, **kwargs)

        return self

    def predict(self, X):
        """
        Predicts the target values for the given input features.

        Parameters
        ----------
        X : array-like
            Input features for prediction.

        Returns
        -------
        numpy.ndarray
            Predicted values for the input features.
        """
        # Convert input features to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.network.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predicted = self.network(X_tensor)  # Forward pass to get predictions
        return predicted.numpy()  # Convert predictions to numpy array

    def score(self, X, y):
        """
        Computes the R2 score of the predictions.

        Parameters
        ----------
        X : array-like
            Input features for scoring.
        y : array-like
            True target values for the input features.

        Returns
        -------
        float
            R2 score indicating the model's performance.
        """
        y_pred = self.predict(X)  # Obtain predictions
        return r2_score(y, y_pred)  # Calculate and return R^2 score

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """
        Returns a list of performance metrics for the predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for the input features.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for the input features.
        list_metrics : list, default=("MSE", "MAE")
            List of metrics to evaluate (can be from Permetrics library: https://github.com/thieu1995/permetrics).

        Returns
        -------
        results : dict
            A dictionary containing the results of the specified metrics.
        """
        return self._BaseAnfis__evaluate_reg(y_true, y_pred, list_metrics)  # Call the evaluation method
