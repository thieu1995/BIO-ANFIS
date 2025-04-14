#!/usr/bin/env python
# Created by "Thieu" at 17:25, 01/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xanfis import MlpClassifier, MlpRegressor


def check_MlpClassifier_multi_class():
    # Load and prepare the dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MlpClassifier(hidden_layers=(100, ), act_names="ReLU", dropout_rates=None, act_output=None,
                 epochs=10, batch_size=16, optim="Adam", optim_paras=None,
                 early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                 seed=42, verbose=True)
    model.fit(X_train, y_train)
    res = model.score(X_test, y_test)
    print(res)
    print(model.network)
    print(model.network.get_weights())


def check_MlpClassifier_multi_class_gridsearch():
    # Load and prepare the dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define parameters for grid search
    param_grid = {
        'hidden_layers': [(30,), (50,), [20, 10], (50, 20)],
        'act_names': ["ReLU", "Tanh", "Sigmoid"],
        'dropout_rates': [0.2, 0.3, None],
        'epochs': [10, 20],
        'batch_size': [16, 24],
        'optim': ['Adam', 'SGD'],
        "early_stopping": [True],
        "valid_rate": [0.1, 0.2],
    }

    # Hyperparameter tuning with GridSearchCV
    searcher = GridSearchCV(estimator=MlpClassifier(), param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
    searcher.fit(X_train, y_train)

    # Get best parameters and accuracy
    print("Best Parameters:", searcher.best_params_)
    print("Best Cross-Validation Accuracy:", searcher.best_score_)

    # Evaluate on test set
    best_net = searcher.best_estimator_
    y_pred = best_net.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)

    print(best_net.network)


def check_MlpClassifier_binary_class():
    # Load and prepare the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MlpClassifier(hidden_layers=(100, 50), act_names="ReLU", dropout_rates=0.2, act_output=None,
                 epochs=50, batch_size=8, optim="Adam", optim_paras=None,
                 early_stopping=True, n_patience=5, epsilon=0.0, valid_rate=0.1,
                 seed=42, verbose=True)
    model.fit(X_train, y_train)
    res = model.score(X_test, y_test)
    print(res)
    print(model.network)


def check_MlpClassifier_binary_class_gridsearch():
    # Load and prepare the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define parameters for grid search
    param_grid = {
        'hidden_layers': [(60,), (100,)],
        'act_names': ["ReLU", "Tanh", "Sigmoid"],
        'dropout_rates': [0.2, None],
        'epochs': [10, 20],
        'batch_size': [16, 24],
        'optim': ['Adam', 'SGD'],
        "early_stopping": [True],
        "valid_rate": [0.1, 0.2],
    }

    # Hyperparameter tuning with GridSearchCV
    searcher = GridSearchCV(estimator=MlpClassifier(), param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
    searcher.fit(X_train, y_train)

    # Get best parameters and accuracy
    print("Best Parameters:", searcher.best_params_)
    print("Best Cross-Validation Accuracy:", searcher.best_score_)

    # Evaluate on test set
    best_net = searcher.best_estimator_
    y_pred = best_net.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)

    print(best_net.network)


def check_MlpRegressor_single_output():
    # Load and prepare the dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    model = MlpRegressor(hidden_layers=(30, 15), act_names="ELU", dropout_rates=0.2, act_output=None,
                 epochs=50, batch_size=8, optim="Adam", optim_paras=None,
                 early_stopping=True, n_patience=5, epsilon=0.0, valid_rate=0.1,
                 seed=42, verbose=True)
    model.fit(X_train, y_train)
    res = model.score(X_test, y_test)
    print(res)
    print(model.network)


def check_MlpRegressor_single_output_gridsearch():
    # Load and prepare the dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    # Define parameters for grid search
    param_grid = {
        'hidden_layers': [(60,), (100,)],
        'act_names': ["ELU", "Tanh", "Sigmoid"],
        'dropout_rates': [0.2, None],
        'epochs': [10, 20],
        'batch_size': [16, 24],
        'optim': ['Adam', 'SGD'],
        "early_stopping": [True],
        "valid_rate": [0.1, 0.2],
    }

    # Hyperparameter tuning with GridSearchCV
    searcher = GridSearchCV(estimator=MlpRegressor(), param_grid=param_grid, cv=3,
                            scoring='neg_mean_squared_error', verbose=2)
    searcher.fit(X_train, y_train)

    # Get best parameters and accuracy
    print("Best Parameters:", searcher.best_params_)
    print("Best Cross-Validation Accuracy:", searcher.best_score_)

    # Evaluate on test set
    best_net = searcher.best_estimator_
    # y_pred = best_net.predict(X_test)
    print(best_net.score(X_test, y_test))
    print(best_net.network)


def check_MlpRegressor_multi_output():
    # Load and prepare the dataset
    # 1. Generate synthetic multi-output regression dataset
    # Setting n_targets=2 for multi-output (2 target variables)
    X, y = make_regression(n_samples=1000, n_features=10, n_targets=2, noise=0.1, random_state=42)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    model = MlpRegressor(hidden_layers=(30, 15), act_names="ELU", dropout_rates=0.2, act_output=None,
                 epochs=50, batch_size=8, optim="Adam", optim_paras=None,
                 early_stopping=True, n_patience=5, epsilon=0.0, valid_rate=0.1,
                 seed=42, verbose=True)
    model.fit(X_train, y_train)
    res = model.score(X_test, y_test)
    print(res)
    print(model.network)


def check_MlpRegressor_multi_output_gridsearch():
    # Load and prepare the dataset
    X, y = make_regression(n_samples=1000, n_features=10, n_targets=2, noise=0.1, random_state=42)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # Define parameters for grid search
    param_grid = {
        'hidden_layers': [(60,), (100,)],
        'act_names': ["ELU", "Tanh", "Sigmoid"],
        'dropout_rates': [0.2, None],
        'epochs': [10, 20],
        'batch_size': [16, 24],
        'optim': ['Adam', 'SGD'],
        "early_stopping": [True],
        "valid_rate": [0.1, 0.2],
    }

    # Hyperparameter tuning with GridSearchCV
    searcher = GridSearchCV(estimator=MlpRegressor(), param_grid=param_grid, cv=3,
                            scoring='neg_mean_squared_error', verbose=2)
    searcher.fit(X_train, y_train)

    # Get best parameters and accuracy
    print("Best Parameters:", searcher.best_params_)
    print("Best Cross-Validation Accuracy:", searcher.best_score_)

    # Evaluate on test set
    best_net = searcher.best_estimator_
    # y_pred = best_net.predict(X_test)
    print(best_net.score(X_test, y_test))
    print(best_net.network)


check_MlpClassifier_multi_class()
check_MlpClassifier_multi_class_gridsearch()
check_MlpClassifier_binary_class()
check_MlpClassifier_binary_class_gridsearch()
check_MlpRegressor_single_output()
check_MlpRegressor_single_output_gridsearch()
check_MlpRegressor_multi_output()
check_MlpRegressor_multi_output_gridsearch()
