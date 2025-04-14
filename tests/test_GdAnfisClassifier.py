#!/usr/bin/env python
# Created by "Thieu" at 22:55, 02/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from xanfis import GdAnfisClassifier


@pytest.fixture
def create_model():
    """Fixture to create an instance of GdAnfisClassifier with default parameters."""
    return GdAnfisClassifier(num_rules=10, mf_class="Gaussian", vanishing_strategy="prod",
                             epochs=10, batch_size=16, optim="Adam", seed=42, verbose=False)


@pytest.fixture
def generate_data():
    """Fixture to generate a small synthetic dataset for testing."""
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 2, size=50)
    return X, y


def test_process_data(create_model, generate_data):
    """Test data processing method with validation split."""
    clf = create_model
    X, y = generate_data

    train_loader, X_valid_tensor, y_valid_tensor = clf.process_data(X, y)
    assert train_loader is not None
    assert X_valid_tensor is not None
    assert y_valid_tensor is not None


def test_fit(create_model, generate_data):
    """Test fitting the model on synthetic data."""
    clf = create_model
    X, y = generate_data

    clf.fit(X, y)
    assert clf.size_input == X.shape[1]
    assert clf.size_output == 1  # Assuming binary classification
    assert clf.classes_ is not None


def test_predict(create_model, generate_data):
    """Test predictions on synthetic data."""
    clf = create_model
    X, y = generate_data
    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    assert set(np.unique(y_pred)).issubset(clf.classes_)


def test_score(create_model, generate_data):
    """Test scoring method by checking accuracy against sklearn's accuracy_score."""
    clf = create_model
    X, y = generate_data
    clf.fit(X, y)

    score = clf.score(X, y)
    y_pred = clf.predict(X)
    expected_score = accuracy_score(y, y_pred)
    assert np.isclose(score, expected_score)


def test_predict_proba(create_model, generate_data):
    """Test probability prediction for classification tasks."""
    clf = create_model
    X, y = generate_data
    clf.fit(X, y)

    probas = clf.predict_proba(X)
    assert probas.shape == (len(X), clf.size_output)
    assert np.all((probas >= 0) & (probas <= 1))


def test_evaluate(create_model, generate_data):
    """Test the evaluation method with default metrics."""
    clf = create_model
    X, y = generate_data
    clf.fit(X, y)

    y_pred = clf.predict(X)
    results = clf.evaluate(y, y_pred)
    assert isinstance(results, dict)
    assert "AS" in results  # Assuming "AS" is accuracy score as an example
    assert "RS" in results  # Assuming "RS" is another metric as per your setup


if __name__ == "__main__":
    pytest.main()
