#!/usr/bin/env python
# Created by "Thieu" at 23:03, 02/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from banfis import MhaMlpComparator

# Sample optimizer dictionary for testing
optim_dict = {
    'BaseGA':       {"epoch": 10, "pop_size": 20},
    "OriginalPSO":  {"epoch": 10, "pop_size": 20},
}

# Sample data
X = np.random.rand(100, 5)  # 100 samples, 5 features
y_classification = np.random.randint(0, 2, 100)  # Binary classification target
y_regression = np.random.rand(100)  # Regression target


@pytest.fixture
def classifier_comparator():
    return MhaMlpComparator(
        optim_dict=optim_dict,
        task="classification",
        hidden_layers=(10,),
        act_names="ReLU",
        dropout_rates=None,
        act_output=None,
        obj_name="F1S",
        verbose=True,
        seed=42
    )


@pytest.fixture
def regressor_comparator():
    return MhaMlpComparator(
        optim_dict=optim_dict,
        task="regression",
        hidden_layers=(10,),
        act_names="ReLU",
        dropout_rates=None,
        act_output=None,
        obj_name="R2",
        verbose=True,
        seed=42
    )


def test_initialization(classifier_comparator, regressor_comparator):
    assert classifier_comparator.task == "classification"
    assert regressor_comparator.task == "regression"
    assert len(classifier_comparator.models) == len(optim_dict)
    assert len(regressor_comparator.models) == len(optim_dict)


def test_compare_cross_validate_classification(classifier_comparator):
    results = classifier_comparator.compare_cross_validate(
        X, y_classification, metrics=["AS", "PS", "F1S", "NPV"], cv=3, n_trials=2, to_csv=False
    )
    assert not results.empty
    assert "mean_test_AS" in results.columns
    assert "std_test_AS" in results.columns
    assert "mean_test_F1S" in results.columns
    assert "mean_train_F1S" in results.columns


def test_compare_cross_validate_regression(regressor_comparator):
    results = regressor_comparator.compare_cross_validate(
        X, y_regression, metrics=["MSE", "MAPE", "R2", "KGE", "NSE"], cv=3, n_trials=2, to_csv=False
    )
    assert not results.empty
    assert "mean_test_MSE" in results.columns
    assert "std_test_MSE" in results.columns
    assert "mean_train_MSE" in results.columns
    assert "std_train_MSE" in results.columns


def test_compare_train_test_split(classifier_comparator):
    X_train, y_train = X[:80], y_classification[:80]
    X_test, y_test = X[80:], y_classification[80:]
    results = classifier_comparator.compare_train_test(
        X_train, y_train, X_test, y_test, metrics=["AS", "PS", "F1S", "NPV"], n_trials=2, to_csv=False
    )
    assert not results.empty
    assert "PS_train" in results.columns
    assert "PS_test" in results.columns
    assert "AS_train" in results.columns
    assert "AS_test" in results.columns
    assert "F1S_test" in results.columns
    assert "F1S_train" in results.columns
