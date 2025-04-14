#!/usr/bin/env python
# Created by "Thieu" at 19:05, 14/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from sklearn.datasets import make_regression
from xanfis import AnfisRegressor


@pytest.fixture
def sample_data():
    """Generate synthetic regression data."""
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    return X, y


def test_fit_predict_score(sample_data):
    X, y = sample_data
    model = AnfisRegressor(num_rules=5, epochs=10, batch_size=8, verbose=False)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.reshape(-1, 1).shape

    r2 = model.score(X, y)
    assert isinstance(r2, float)
    assert -1.0 <= r2 <= 1.0


def test_multi_output_fit_predict():
    X = np.random.rand(100, 3)
    y = np.random.rand(100, 2)
    model = AnfisRegressor(num_rules=4, epochs=5, batch_size=10, verbose=False)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_evaluate_metrics(sample_data):
    X, y = sample_data
    model = AnfisRegressor(num_rules=5, epochs=10, batch_size=8, verbose=False)
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = model.evaluate(y, y_pred, list_metrics=["MSE", "MAE"])
    assert isinstance(metrics, dict)
    assert "MSE" in metrics and "MAE" in metrics
    assert isinstance(metrics["MSE"], float)
    assert isinstance(metrics["MAE"], float)


def test_process_data_validation(sample_data):
    X, y = sample_data
    model = AnfisRegressor(valid_rate=0.2, batch_size=10, verbose=False)
    train_loader, X_val_tensor, y_val_tensor = model.process_data(X, y)

    assert train_loader is not None
    assert X_val_tensor is not None and y_val_tensor is not None
    assert X_val_tensor.shape[0] == y_val_tensor.shape[0]


def test_invalid_validation_rate(sample_data):
    X, y = sample_data
    with pytest.raises(ValueError):
        model = AnfisRegressor(valid_rate=1.5)
        model.process_data(X, y)


def test_predict_without_fit(sample_data):
    X, y = sample_data
    model = AnfisRegressor()
    with pytest.raises(AttributeError):
        model.predict(X)
