#!/usr/bin/env python
# Created by "Thieu" at 22:45, 02/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from banfis import MhaMlpRegressor


@pytest.fixture
def regression_data():
    # Create a synthetic dataset for regression
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regressor():
    # Initialize the MhaMlpRegressor with default parameters
    return MhaMlpRegressor(
        hidden_layers=[50, 25],
        act_names="ReLU",
        dropout_rates=0.2,
        optim="BaseGA",
        optim_paras={"epoch": 10, "pop_size": 20},
        seed=42,
        verbose=False
    )


def test_initialization(regressor):
    # Test if the regressor initializes correctly
    assert isinstance(regressor, MhaMlpRegressor)
    assert regressor.seed == 42


def test_fit(regressor, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # Test if the regressor can fit the model
    regressor.fit(X_train, y_train)
    assert regressor.task in ["regression", "multi_regression"]  # Task should be set
    assert hasattr(regressor, "network")  # Model should be built after fitting


def test_predict(regressor, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # Train the model before predicting
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    # Check if predictions have the same length as test samples
    assert len(predictions) == len(y_test)


def test_score(regressor, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # Train the model and calculate R^2 score
    regressor.fit(X_train, y_train)
    r2 = regressor.score(X_test, y_test)

    # Compare with sklearn's R^2 score
    predictions = regressor.predict(X_test)
    expected_r2 = r2_score(y_test, predictions)
    assert r2 == pytest.approx(expected_r2, 0.01)


def test_evaluate(regressor, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # Train the model and get predictions
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    # Evaluate with custom metrics
    metrics = regressor.evaluate(y_test, predictions, list_metrics=("MSE", "MAE"))

    # Check if metrics dictionary contains requested metrics
    assert "MSE" in metrics
    assert "MAE" in metrics
    assert isinstance(metrics["MSE"], float)
    assert isinstance(metrics["MAE"], float)
