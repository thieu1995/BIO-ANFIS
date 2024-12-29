#!/usr/bin/env python
# Created by "Thieu" at 10:42, 17/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from banfis import MhaMlpRegressor, MhaMlpClassifier
from banfis.helpers.metric_util import get_metric_sklearn


class MhaMlpComparator:
    """
    Automated compare different MhaMlp models based on provided optimizer configurations.

    This class facilitates the comparison of multiple MhaMlp models with varying optimizer
    configurations. It provides methods for cross-validation and train-test split evaluation.

    Args:
        optim_dict (dict, optional): A dictionary of optimizer names and parameters.
        task (str, optional): The task type (classification or regression). Defaults to 'classification'.
        hidden_layers (int, list, tuple, optional): The number of nodes in each hidden layer. Defaults is (10, ).
        act_names (str, optional): The activation function name. Defaults to 'ELU'.
        obj_name (str, optional): The objective function name. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        obj_weights (array-like, optional): Weights for the objective function. Defaults to None.
        **kwargs: Additional keyword arguments for model initialization.
    """

    def __init__(self, optim_dict=None, task="classification", hidden_layers=(10, ), act_names="ELU",
                 dropout_rates=None, act_output=None, obj_name=None, verbose=False, seed=None, **kwargs):
        self.optim_dict = self._set_optimizer_dict(optim_dict)
        self.hidden_layers = hidden_layers
        self.act_names = act_names
        self.dropout_rates = dropout_rates
        self.act_output = act_output
        self.obj_name = obj_name
        self.verbose = verbose
        self.generator = np.random.default_rng(seed)
        self.task = task
        if self.task == "classification":
            self.models = [MhaMlpClassifier(hidden_layers=hidden_layers, act_names=act_names,
                                            dropout_rates=dropout_rates, act_output=act_output, obj_name=obj_name,
                                            verbose=verbose, seed=seed) for _ in range(len(self.optim_dict))]
        else:
            self.models = [MhaMlpRegressor(hidden_layers=hidden_layers, act_names=act_names,
                                            dropout_rates=dropout_rates, act_output=act_output, obj_name=obj_name,
                                            verbose=verbose, seed=seed) for _ in range(len(self.optim_dict))]
        self.kwargs = kwargs
        self.best_estimator_ = None
        self.best_params_ = None
        self.result_cross_val_scores_ = None
        self.result_train_test_ = None

    def _set_optimizer_dict(self, opt_dict):
        """Validates the optimizer dictionary."""
        if type(opt_dict) is not dict:
            raise TypeError(f"Support optim_dict hyper-parameter as dict only: {type(opt_dict)}")
        return opt_dict

    def _filter_metric_results(self, results, metric_names=None, return_train_score=False):
        """Filters and formats metric results."""
        list_metrics = []
        for idx, metric in enumerate(metric_names):
            list_metrics.append(f"test_{metric}")
            if return_train_score:
                list_metrics.append(f"train_{metric}")
        final_results = {}
        for idx, metric in enumerate(list_metrics):
            final_results[f"mean_{metric}"] = np.mean(results[metric])
            final_results[f"std_{metric}"] = np.std(results[metric])
        return final_results

    def _rename_metrics(self, results: dict, suffix="train"):
        """Renames metric keys with a specified suffix."""
        res = {}
        for metric_name, metric_value in results.items():
            res[f"{metric_name}_{suffix}"] = metric_value
        return res

    def _results_to_csv(self, to_csv=False, results=None, saved_file_path="history/results.csv"):
        """Saves results to a CSV file."""
        Path(f"{Path.cwd()}/{Path(saved_file_path).parent}").mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        if to_csv:
            if not saved_file_path.lower().endswith(".csv"):
                saved_file_path += ".csv"
            df.to_csv(saved_file_path, index=False)
        return df

    def compare_cross_validate(self, X, y, metrics=None, cv=5, return_train_score=True, n_trials=10,
                               to_csv=True, saved_file_path="history/results_cross_validate.csv", **kwargs):
        """Performs cross-validation for model comparison.

        Compares different MhaMlp models using cross-validation.

        Args:
            X (array-like): The feature matrix.
            y (array-like): The target vector.
            metrics (list, optional): A list of metric names. Defaults to None.
            cv (int, optional): The number of cross-validation folds. Defaults to 5.
            return_train_score (bool, optional): Whether to return train scores. Defaults to True.
            n_trials (int, optional): The number of trials. Defaults to 10.
            to_csv (bool, optional): Whether to save results to a CSV file. Defaults to True.
            saved_file_path (str, optional): The path to save the CSV file. Defaults to 'history/results_cross_validate.csv'.
            **kwargs: Additional keyword arguments for cross_validate.

        Returns:
            pandas.DataFrame: The comparison results.
        """
        list_seeds = self.generator.choice(list(range(0, 1000)), n_trials, replace=False)
        scoring = get_metric_sklearn(task=self.task, metric_names=metrics)
        results = []
        for idx, (opt_name, opt_paras) in enumerate(self.optim_dict.items()):
            self.models[idx].set_optim_and_paras(opt_name, opt_paras)
            for trial, seed in enumerate(list_seeds):
                self.models[idx].set_seed(seed)
                res = cross_validate(self.models[idx], X, y, scoring=scoring, cv=cv, return_train_score=return_train_score, **kwargs)
                final_res = self._filter_metric_results(res, list(scoring.keys()), return_train_score)
                temp = {"model_name": f"{opt_name}-{opt_paras}", "trial": trial, **final_res}
                results.append(temp)
        return self._results_to_csv(to_csv, results=results, saved_file_path=saved_file_path)

    def compare_cross_val_score(self, X, y, metric=None, cv=5, n_trials=10, to_csv=True,
                                saved_file_path="history/results_cross_val_score.csv", **kwargs):
        """Performs cross-validation with a single metric.

        Compares different MhaMlp models using cross-validation with a single metric.

        Args:
            X (array-like): The feature matrix.
            y (array-like): The target vector.
            metric (str, optional): The metric to evaluate. Defaults to None.
            cv (int, optional): The number of cross-validation folds. Defaults to 5.
            n_trials (int, optional): The number of trials. Defaults to 10.
            to_csv (bool, optional): Whether to save results to a CSV file. Defaults to True.
            saved_file_path (str, optional): The path to save the CSV file. Defaults to 'history/results_cross_val_score.csv'.
            **kwargs: Additional keyword arguments for cross_val_score.

        Returns:
            pandas.DataFrame: The comparison results.
        """
        list_seeds = self.generator.choice(list(range(0, 1000)), n_trials, replace=False)
        scoring = get_metric_sklearn(task=self.task, metric_names=[metric])[metric]
        results = []
        for idx, (opt_name, opt_paras) in enumerate(self.optim_dict.items()):
            self.models[idx].set_optim_and_paras(opt_name, opt_paras)
            for trial, seed in enumerate(list_seeds):
                self.models[idx].set_seed(seed)
                res = cross_val_score(self.models[idx], X, y, scoring=scoring, cv=cv, **kwargs)
                final_res = {"model_name": f"{opt_name}-{opt_paras}", "trial": trial, f"mean_{metric}": np.mean(res), f"std_{metric}": np.std(res)}
                results.append(final_res)
        return self._results_to_csv(to_csv, results=results, saved_file_path=saved_file_path)

    def compare_train_test(self, X_train, y_train, X_test, y_test, metrics=None, n_trials=10,
                           to_csv=True, saved_file_path="history/results_train_test.csv"):
        """Compares models using train-test split.

        Compares different MhaMlp models using train-test split evaluation.

        Args:
            X_train (array-like): The training feature matrix.
            y_train (array-like): The training target vector.
            X_test (array-like): The testing feature matrix.
            y_test (array-like): The testing target vector.
            metrics (list, optional): A list of metric names. Defaults to None.
            n_trials (int, optional): The number of trials. Defaults to 10.
            to_csv (bool, optional): Whether to save results to a CSV file. Defaults to True.
            saved_file_path (str, optional): The path to save the CSV file. Defaults to 'history/results_train_test.csv'.

        Returns:
            pandas.DataFrame: The comparison results.
        """
        list_seeds = self.generator.choice(list(range(0, 1000)), n_trials, replace=False)
        for idx, (opt_name, opt_paras) in enumerate(self.optim_dict.items()):
            self.models[idx].set_optim_and_paras(opt_name, opt_paras)
        results = []
        for idx, _ in enumerate(self.models):
            for trial, seed in enumerate(list_seeds):
                self.models[idx].set_seed(seed)
                self.models[idx].fit(X_train, y_train)
                model_name = self.models[idx].get_name()
                y_train_pred = self.models[idx].predict(X_train)
                res1 = self.models[idx].evaluate(y_train, y_train_pred, metrics)
                y_test_pred = self.models[idx].predict(X_test)
                res2 = self.models[idx].evaluate(y_test, y_test_pred, metrics)
                res1 = self._rename_metrics(res1, suffix="train")
                res2 = self._rename_metrics(res2, suffix="test")
                if self.verbose:
                    temp = {**res1, **res2}
                    print(f"{model_name} model is trained and evaluated with score: {temp}")
                results.append({"model": model_name, "trial": trial, **res1, **res2})
        return self._results_to_csv(to_csv, results=results, saved_file_path=saved_file_path)
