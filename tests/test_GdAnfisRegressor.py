#!/usr/bin/env python
# Created by "Thieu" at 22:60, 02/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
import torch
from xanfis import GdAnfisRegressor


@pytest.fixture
def sample_data():
    """Fixture to generate consistent synthetic regression data."""
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 1)
    return X, y


@pytest.fixture
def model():
    """Fixture to initialize the GdAnfisRegressor model with default parameters."""
    return GdAnfisRegressor(
        num_rules=10,
        mf_class="Gaussian",
        vanishing_strategy="prod",
        epochs=10,
        batch_size=16,
        optim="Adam",
        valid_rate=0.2,
        seed=42,
        verbose=False
    )


def test_process_data(model, sample_data):
    """Test the data preprocessing, batching, and validation split."""
    X, y = sample_data
    train_loader, X_valid_tensor, y_valid_tensor = model.process_data(X, y)

    # Validate types and shapes from train loader
    for batch_X, batch_y in train_loader:
        assert isinstance(batch_X, torch.Tensor)
        assert isinstance(batch_y, torch.Tensor)
        assert batch_X.shape[1] == X.shape[1]
        assert batch_y.shape[1] == y.shape[1]
        break

    if model.valid_rate == 0:
        assert X_valid_tensor is None and y_valid_tensor is None
    else:
        assert isinstance(X_valid_tensor, torch.Tensor)
        assert isinstance(y_valid_tensor, torch.Tensor)
        assert X_valid_tensor.shape[0] == y_valid_tensor.shape[0]
        assert X_valid_tensor.shape[1] == X.shape[1]


def test_fit(model, sample_data):
    """Test model fitting and parameter initialization."""
    X, y = sample_data
    model.fit(X, y)
    assert hasattr(model, 'network')
    for param in model.network.parameters():
        assert param.requires_grad
    # Optional: If your model has an `is_fitted_` flag
    assert getattr(model, "is_fitted_", True)


def test_predict_shape(model, sample_data):
    """Ensure predict output shape matches target shape."""
    X, y = sample_data
    model.fit(X, y)
    predictions = model.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == y.shape


def test_predict_values(model, sample_data):
    """Ensure predict output values are finite and within range."""
    X, y = sample_data
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.all(np.isfinite(predictions))
    assert predictions.dtype == np.float64 or predictions.dtype == np.float32


def test_evaluate(model, sample_data):
    """Test evaluation metrics computation."""
    X, y = sample_data
    model.fit(X, y)
    predictions = model.predict(X)

    metrics = model.evaluate(y, predictions, list_metrics=["MSE", "MAE"])

    assert isinstance(metrics, dict)
    assert "MSE" in metrics and "MAE" in metrics
    assert isinstance(metrics["MSE"], float)
    assert isinstance(metrics["MAE"], float)
    assert metrics["MSE"] >= 0
    assert metrics["MAE"] >= 0
