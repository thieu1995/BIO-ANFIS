#!/usr/bin/env python
# Created by "Thieu" at 22:60, 02/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
import torch
from banfis import MlpRegressor


# Test data
X = np.random.rand(100, 5)  # 100 samples, 5 features each
y = np.random.rand(100, 1)  # 100 target values


@pytest.fixture
def model():
    """Fixture to initialize the MlpRegressor model with default parameters."""
    return MlpRegressor(hidden_layers=(50, 50), epochs=10, batch_size=8, seed=42, verbose=False)


def test_initialization(model):
    """Test that the MlpRegressor initializes with correct default parameters."""
    assert model.hidden_layers == (50, 50)
    assert model.epochs == 10
    assert model.batch_size == 8
    assert model.seed == 42
    assert model.verbose is False


def test_process_data(model):
    """Test the data processing and tensor conversion."""
    train_loader, X_valid_tensor, y_valid_tensor = model.process_data(X, y)

    # Check the training loader data format
    for batch_X, batch_y in train_loader:
        assert isinstance(batch_X, torch.Tensor)
        assert isinstance(batch_y, torch.Tensor)
        break  # Check only the first batch

    # Check if validation tensors are None when valid_rate is not set
    if model.valid_rate == 0:
        assert X_valid_tensor is None
        assert y_valid_tensor is None
    else:
        assert X_valid_tensor is not None
        assert y_valid_tensor is not None


def test_fit(model):
    """Test the fitting process of the model."""
    # Fit the model
    model.fit(X, y)

    # Check if the model is trained (weights initialized)
    assert hasattr(model, 'network')
    for param in model.network.parameters():
        assert param.requires_grad


def test_predict_shape(model):
    """Test that the predict method outputs the correct shape."""
    model.fit(X, y)  # Fit before predicting
    predictions = model.predict(X)
    assert predictions.shape == y.shape  # Shape of predictions should match shape of y


def test_predict_values(model):
    """Test that predict outputs reasonable values for a trained model."""
    model.fit(X, y)
    predictions = model.predict(X)
    # Assert predictions are finite numbers
    assert np.all(np.isfinite(predictions))


def test_score_r2(model):
    """Test the scoring method to verify R2 score calculation."""
    model.fit(X, y)
    r2 = model.score(X, y)
    assert isinstance(r2, float)
    assert -1 <= r2 <= 1  # R2 score should be within [-1, 1]


def test_evaluate(model):
    """Test the evaluate method to verify performance metrics."""
    model.fit(X, y)
    predictions = model.predict(X)

    # Evaluate with Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    results = model.evaluate(y, predictions, list_metrics=["MSE", "MAE"])

    # Check if results contain the expected keys
    assert "MSE" in results
    assert "MAE" in results
    assert results["MSE"] >= 0  # MSE should be non-negative
    assert results["MAE"] >= 0  # MAE should be non-negative
