#!/usr/bin/env python
# Created by "Thieu" at 13:43, 13/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from permetrics import ClassificationMetric, RegressionMetric
from sklearn.base import BaseEstimator
from banfis.helpers import validator
from banfis.helpers.metric_util import get_all_classification_metrics, get_all_regression_metrics


class MlpTorch(nn.Module):
    """
    Define the MLP model
    """
    SUPPORTED_ACTIVATIONS = ['none', 'threshold', 'relu', 'rrelu', 'hardtanh', 'relu6', 'sigmoid',
                             'hardsigmoid', 'tanh', 'silu', 'mish', 'hardswish', 'elu', 'celu',
                             'selu', 'glu', 'gelu', 'hardshrink', 'leakyrelu', 'logsigmoid',
                             'softplus', 'softshrink', 'multiheadattention', 'prelu', 'softsign',
                             'tanhshrink', 'softmin', 'softmax', 'logsoftmax']

    def __init__(self, input_size, hidden_size, output_size, act1_name="tanh", act2_name="sigmoid"):
        super(MlpTorch, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        self.act1_func = self.__get_act(act1_name)
        self.act2_func = self.__get_act(act2_name)

    def __get_act(self, act_name):
        if act_name == "softmax":
            act_func = nn.Softmax(dim=0)
        elif act_name == "none":
            act_func = nn.Identity()
        else:
            act_func = getattr(nn.functional, act_name)
        return act_func

    def forward(self, x):
        x = x.clone().detach()  # torch.tensor(x, dtype=torch.float32)
        x = self.fc1(x)
        x = self.act1_func(x)
        x = self.fc2(x)
        x = self.act2_func(x)
        return x


class BaseMlpTorch(BaseEstimator):
    """
    Defines the most general class for traditional MLP models that inherits the BaseEstimator class of Scikit-Learn library.

    Parameters
    ----------
    hidden_size : int, default=50
        The hidden size of MLP network (This network only has single hidden layer).

    act1_name : str, defeault="tanh"
        This is activation for hidden layer. The supported activation are: {"none", "relu",
        "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus",
        "mish", "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax"}.

    act2_name : str, defeault="sigmoid"
        This is activation for output layer. The supported activation are:
        {"none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish",
        "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax"}.

    obj_name : str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

    max_epochs : int, default=1000
        Maximum number of epochs / iterations / generations

    batch_size : int, default=32
        The batch size

    optimizer : str, default = "SGD"
        The gradient-based optimizer from Pytorch. List of supported optimizer is:
        ["Adadelta", "Adagrad", "Adam", "Adamax", "AdamW", "ASGD", "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"]

    optimizer_paras : dict or None, default=None
        The dictionary parameters of the selected optimizer.

    verbose : bool, default=True
        Whether to print progress messages to stdout.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = None

    SUPPORTED_LOSSES = {
        "MAE": torch.nn.L1Loss, "MSE": torch.nn.MSELoss
    }
    SUPPORTED_OPTIMIZERS = ["Adadelta", "Adagrad", "Adam", "Adamax", "AdamW", "ASGD",
                            "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"]

    def __init__(self, hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name=None,
                 max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False):
        super().__init__()
        self.module = MlpTorch
        self.hidden_size = validator.check_int("hidden_size", hidden_size, [2, 1000000])
        self.act1_name = validator.check_str("act1_name", act1_name, MlpTorch.SUPPORTED_ACTIVATIONS)
        self.act2_name = validator.check_str("act2_name", act2_name, MlpTorch.SUPPORTED_ACTIVATIONS)
        self.obj_name = validator.check_str("obj_name", obj_name, list(self.SUPPORTED_LOSSES.keys()))
        self.max_epochs = validator.check_int("max_epochs", max_epochs, [1, 1000000])
        self.batch_size = batch_size
        self.optimizer = validator.check_str("optimizer", optimizer, self.SUPPORTED_OPTIMIZERS)
        self.optimizer_paras = {} if optimizer_paras is None else optimizer_paras
        self.verbose = verbose

        self.weights = {}
        self.network, self.obj_scaler, self.loss_train = None, None, None
        self.n_labels, self.obj_scaler = None, None

    @staticmethod
    def _check_method(method=None, list_supported_methods=None):
        if type(method) is str:
            return validator.check_str("method", method, list_supported_methods)
        else:
            raise ValueError(f"method should be a string and belongs to {list_supported_methods}")

    def create_network(self, X, y):
        return None, None

    def fit(self, X, y):
        pass

    def predict(self, X, return_prob=False):
        """
        Inherit the predict function from BaseMlp class, with 1 more parameter `return_prob`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        return_prob : bool, default=False
            It is used for classification problem:

            - If True, the returned results are the probability for each sample
            - If False, the returned results are the predicted labels
        """
        X = torch.tensor(X, dtype=torch.float32)
        if return_prob:
            return self.network.predict_proba(X)
        else:
            return self.network.predict(X)

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred, decimal=8)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        cm = ClassificationMetric(y_true, y_pred, decimal=8)
        return cm.get_metrics_by_list_names(list_metrics)

    def __score_reg(self, X, y, method="RMSE"):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        method = self._check_method(method, list(self.SUPPORTED_REG_METRICS.keys()))
        y_pred = self.network.predict(X)
        return RegressionMetric(y, y_pred, decimal=6).get_metric_by_name(method)[method]

    def __scores_reg(self, X, y, list_methods=("MSE", "MAE")):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        y_pred = self.network.predict(X)
        rm = RegressionMetric(y_true=y, y_pred=y_pred, decimal=6)
        return rm.get_metrics_by_list_names(list_methods)

    def __score_cls(self, X, y, method="AS"):
        """
        Return the metric on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        method : str, default="AS"
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        method = self._check_method(method, list(self.SUPPORTED_CLS_METRICS.keys()))
        return_prob = False
        if self.n_labels > 2:
            if method in self.CLS_OBJ_LOSSES:
                return_prob = True
        y_pred = self.predict(X, return_prob=return_prob)
        cm = ClassificationMetric(y_true=y, y_pred=y_pred, decimal=6)
        return cm.get_metric_by_name(method)[method]

    def __scores_cls(self, X, y, list_methods=("AS", "RS")):
        """
        Return the list of metrics on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        list_methods : list, default=("AS", "RS")
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        list_errors = list(set(list_methods) & set(self.CLS_OBJ_LOSSES))
        list_scores = list((set(self.SUPPORTED_CLS_METRICS.keys()) - set(self.CLS_OBJ_LOSSES)) & set(list_methods))
        t1 = {}
        if len(list_errors) > 0:
            return_prob = False
            if self.n_labels > 2:
                return_prob = True
            y_pred = self.predict(X, return_prob=return_prob)
            cm = ClassificationMetric(y, y_pred, decimal=6)
            t1 = cm.get_metrics_by_list_names(list_errors)
        y_pred = self.predict(X, return_prob=False)
        cm = ClassificationMetric(y, y_pred, decimal=6)
        t2 = cm.get_metrics_by_list_names(list_scores)
        return {**t2, **t1}

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        pass

    def score(self, X, y, method=None):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        pass

    def scores(self, X, y, list_methods=None):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        pass

    def save_training_loss(self, save_path="history", filename="loss.csv"):
        """
        Save the loss (convergence) during the training process to csv file.

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} model doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_evaluation_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to csv file

        Parameters
        ----------
        y_true : ground truth data
        y_pred : predicted output
        list_metrics : list of evaluation metrics
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save the predicted results to csv file

        Parameters
        ----------
        X : The features data, nd.ndarray
        y_true : The ground truth data
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X, return_prob=False)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="model.pkl"):
        """
        Save model to pickle file

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".pkl" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="model.pkl"):
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))
