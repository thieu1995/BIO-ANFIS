#!/usr/bin/env python
# Created by "Thieu" at 23:23, 02/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.exceptions import NotFittedError
from banfis import MhaMlpTuner


@pytest.fixture
def sample_data():
    """Creates sample data for testing"""
    X = np.random.rand(20, 5)
    y_classification = np.random.randint(0, 2, size=20)
    y_regression = np.random.rand(20)
    return X, y_classification, y_regression


@pytest.fixture
def param_dict():
    """Creates a sample parameter grid"""
    return {
        'hidden_layers': [(10,), ],
        'act_names': ['Tanh', 'ELU'],
        'dropout_rates': [None],
        'optim': ['BaseGA'],
        'optim_paras': [
            {"epoch": 10, "pop_size": 20},
            {"epoch": 20, "pop_size": 20},
        ],
        'obj_name': ["F1S"],
        'seed': [42],
        "verbose": [False],
    }


def test_init_invalid_task():
    """Test initialization with an invalid task"""
    with pytest.raises(ValueError, match="Invalid task type"):
        MhaMlpTuner(task="invalid_task")


def test_get_search_object_invalid_method(param_dict):
    """Test _get_search_object with invalid search method"""
    tuner = MhaMlpTuner(task="classification", param_dict=param_dict, search_method="invalid_method",
                        scoring="accuracy")
    with pytest.raises(ValueError, match="Unsupported searching method"):
        tuner._get_search_object()


def test_get_search_object_no_param_dict():
    """Test _get_search_object without param_dict"""
    tuner = MhaMlpTuner(task="classification", scoring="accuracy")
    with pytest.raises(ValueError, match="requires a param_dict"):
        tuner._get_search_object()


def test_get_search_object_no_scoring(param_dict):
    """Test _get_search_object without scoring"""
    tuner = MhaMlpTuner(task="classification", param_dict=param_dict)
    with pytest.raises(ValueError, match="requires a scoring method"):
        tuner._get_search_object()


def test_fit_classification_grid_search(sample_data, param_dict):
    """Test fitting with classification task and GridSearchCV"""
    X, y_classification, _ = sample_data
    tuner = MhaMlpTuner(task="classification", param_dict=param_dict, search_method="gridsearch", scoring="accuracy")
    tuner.fit(X, y_classification)
    assert isinstance(tuner.searcher, GridSearchCV)
    assert tuner.best_estimator_ is not None
    assert tuner.best_params_ is not None


def test_fit_regression_random_search(sample_data, param_dict):
    """Test fitting with regression task and RandomizedSearchCV"""
    X, _, y_regression = sample_data
    param_dict["obj_name"] = ["MSE"]
    tuner = MhaMlpTuner(task="regression", param_dict=param_dict, search_method="randomsearch",
                        scoring="neg_mean_squared_error", n_iter=2)
    tuner.fit(X, y_regression)
    assert isinstance(tuner.searcher, RandomizedSearchCV)
    assert tuner.best_estimator_ is not None
    assert tuner.best_params_ is not None


def test_predict_before_fit(sample_data, param_dict):
    """Test predict method before calling fit"""
    X, _, _ = sample_data
    tuner = MhaMlpTuner(task="classification", param_dict=param_dict, scoring="accuracy")
    with pytest.raises(NotFittedError, match="not fitted yet"):
        tuner.predict(X)


def test_predict_after_fit(sample_data, param_dict):
    """Test predict method after fitting"""
    X, y_classification, _ = sample_data
    tuner = MhaMlpTuner(task="classification", param_dict=param_dict, search_method="gridsearch", scoring="accuracy")
    tuner.fit(X, y_classification)
    predictions = tuner.predict(X)
    assert predictions.shape == y_classification.shape


def test_best_params_after_fit(sample_data, param_dict):
    """Check if best_params_ is set after fitting"""
    X, y_classification, _ = sample_data
    tuner = MhaMlpTuner(task="classification", param_dict=param_dict, search_method="gridsearch", scoring="accuracy")
    tuner.fit(X, y_classification)
    assert tuner.best_params_ is not None
