#!/usr/bin/env python
# Created by "Thieu" at 22:40, 02/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xanfis import BioAnfisClassifier


@pytest.fixture
def data():
    # Create a synthetic dataset for classification
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def classifier():
    # Initialize the BioAnfisClassifier with default parameters
    return BioAnfisClassifier(
        num_rules=10, mf_class="Gaussian", vanishing_strategy="prod",
        optim="BaseGA", optim_params={"epoch": 10, "pop_size": 20},
        seed=42, verbose=False
    )


def test_initialization(classifier):
    # Test if the classifier initializes correctly
    assert isinstance(classifier, BioAnfisClassifier)
    assert classifier.seed == 42
    assert classifier.task == "classification"  # Default should be classification


def test_fit(classifier, data):
    X_train, X_test, y_train, y_test = data
    # Test if the classifier can fit the model
    classifier.fit(X_train, y_train)
    assert classifier.classes_ is not None
    assert len(classifier.classes_) == 2


def test_predict(classifier, data):
    X_train, X_test, y_train, y_test = data
    # Train the model before predicting
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # Check if predictions have the same length as test samples
    assert len(predictions) == len(y_test)


def test_predict_proba(classifier, data):
    X_train, X_test, y_train, y_test = data
    # Train the model before predicting probabilities
    classifier.fit(X_train, y_train)

    # Check if predict_proba returns probabilities
    probs = classifier.predict_proba(X_test)
    assert probs.shape[0] == len(X_test)
    assert probs.shape[1] == (1 if classifier.task == "binary_classification" else len(classifier.classes_))


def test_score(classifier, data):
    X_train, X_test, y_train, y_test = data
    # Train the model and calculate accuracy
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)

    # Compare with sklearn's accuracy_score
    predictions = classifier.predict(X_test)
    expected_accuracy = accuracy_score(y_test, predictions)
    assert accuracy == pytest.approx(expected_accuracy, 0.01)


def test_evaluate(classifier, data):
    X_train, X_test, y_train, y_test = data
    # Train the model and get predictions
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # Evaluate with custom metrics
    metrics = classifier.evaluate(y_test, predictions, list_metrics=("AS", "RS"))

    # Check if metrics dictionary contains requested metrics
    assert "AS" in metrics
    assert "RS" in metrics
    assert isinstance(metrics["AS"], float)
    assert isinstance(metrics["RS"], float)
