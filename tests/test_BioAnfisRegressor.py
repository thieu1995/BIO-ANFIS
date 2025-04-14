#!/usr/bin/env python
# Created by "Thieu" at 19:13, 14/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from xanfis import BioAnfisRegressor


# Synthetic dataset
X = np.random.rand(80, 4)
y = np.random.rand(80, 1)


@pytest.fixture
def model():
    """Fixture to create a BioAnfisRegressor instance."""
    return BioAnfisRegressor(
        num_rules=10, mf_class="Gaussian", vanishing_strategy="prod",
        optim="BaseGA", optim_params={"epoch": 10, "pop_size": 20},
        seed=42, verbose=False
    )


def test_fit_single_output(model):
    """Test the fit method on single-output regression."""
    model.fit(X, y)
    assert hasattr(model, "network")
    assert model.task == "regression"
    assert model.size_input == X.shape[1]
    assert model.size_output == 1


def test_fit_multi_output():
    """Test the fit method on multi-output regression."""
    X_multi = np.random.rand(50, 3)
    y_multi = np.random.rand(50, 2)
    model = BioAnfisRegressor(num_rules=10, mf_class="Gaussian",
                              optim="BaseGA", optim_params={"epoch": 10, "pop_size": 20},
                              seed=0, verbose=False)
    model.fit(X_multi, y_multi)
    assert model.task == "multi_regression"
    assert model.size_output == 2


def test_predict_shape(model):
    """Ensure the predict output shape matches the target shape."""
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_values_are_finite(model):
    """Ensure predictions are finite numbers."""
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_score_r2(model):
    """Test that score returns a valid R^2."""
    model.fit(X, y)
    r2 = model.score(X, y)
    assert isinstance(r2, float)
    assert -1.0 <= r2 <= 1.0


def test_evaluate_metrics(model):
    """Test the evaluate method with default metrics."""
    model.fit(X, y)
    preds = model.predict(X)
    results = model.evaluate(y, preds, list_metrics=["MSE", "MAE"])

    assert isinstance(results, dict)
    assert "MSE" in results
    assert "MAE" in results
    assert all(np.isfinite(val) for val in results.values())
