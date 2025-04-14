#!/usr/bin/env python
# Created by "Thieu" at 18:08, 14/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xanfis import AnfisClassifier


@pytest.fixture
def synthetic_dataset():
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                               n_redundant=0, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_fit_and_predict(synthetic_dataset):
    X_train, X_test, y_train, y_test = synthetic_dataset

    clf = AnfisClassifier(num_rules=5, mf_class="Gaussian", vanishing_strategy="prod",
                          epochs=5, batch_size=16, verbose=False, seed=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_test.shape
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_score_function(synthetic_dataset):
    X_train, X_test, y_train, y_test = synthetic_dataset

    clf = AnfisClassifier(num_rules=5, mf_class="Gaussian", vanishing_strategy="prod",
                          epochs=5, batch_size=16, verbose=False, seed=42)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_proba(synthetic_dataset):
    X_train, X_test, y_train, y_test = synthetic_dataset

    clf = AnfisClassifier(num_rules=5, mf_class="Gaussian", vanishing_strategy="prod",
                          epochs=5, batch_size=16, verbose=False, seed=42)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)

    assert isinstance(proba, np.ndarray)
    assert proba.shape == (X_test.shape[0], 1) or proba.shape == (X_test.shape[0], 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_invalid_validation_rate():
    with pytest.raises(ValueError):
        _ = AnfisClassifier(valid_rate=1.5).fit(np.random.rand(10, 4), np.random.randint(0, 2, size=10))
