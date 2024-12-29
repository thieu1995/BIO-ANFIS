#!/usr/bin/env python
# Created by "Thieu" at 09:10, 01/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from banfis import MlpClassifier, MlpRegressor, MhaMlpClassifier, MhaMlpRegressor


def check_MhaMlpClassifier_multi_class():
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

    model = MhaMlpClassifier(hidden_layers=(100, ), act_names="ELU", dropout_rates=None, act_output=None,
                           optim="BaseGA", optim_paras=None, obj_name="F1S", seed=42, verbose=True)
    # print(model.SUPPORTED_CLS_OBJECTIVES)

    model.fit(X_train, y_train, mode="swarm", n_workers=6)
    res = model.score(X_test, y_test)
    print(res)
    print(model.network)
    print(model.network.get_weights())


def check_MhaMlpClassifier_multi_class_gridsearch():
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
        'hidden_layers': [(30,), [20, 10], (50, 20)],
        'act_names': ["ReLU", "Tanh", "Sigmoid"],
        'dropout_rates': [0.2, 0.3, None],
        'optim': ['BaseGA', 'OriginalWOA'],
        "optim_paras": [
            {"epoch": 10, "pop_size": 30 },
            {"epoch": 20, "pop_size": 30 },
        ],
        "obj_name": ["F1S"]
    }

    # Hyperparameter tuning with GridSearchCV
    searcher = GridSearchCV(estimator=MhaMlpClassifier(), param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
    searcher.fit(X_train, y_train)

    # Get best parameters and accuracy
    print("Best Parameters:", searcher.best_params_)
    print("Best Cross-Validation Accuracy:", searcher.best_score_)

    # Evaluate on test set
    best_model = searcher.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)

    print(best_model.network)


def check_MhaMlpClassifier_binary_class():
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

    model = MhaMlpClassifier(hidden_layers=(100,), act_names="ELU", dropout_rates=0.2, act_output=None,
                     optim="BaseGA", optim_paras=None, obj_name="F1S", seed=42, verbose=True)
    model.fit(X_train, y_train)
    res = model.score(X_test, y_test)
    print(res)
    print(model.network)


def check_MhaMlpClassifier_binary_class_gridsearch():
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
        'hidden_layers': [(30,), (50, 20)],
        'act_names': ["ReLU", "Tanh", "Sigmoid"],
        'dropout_rates': [0.2, None],
        'optim': ['BaseGA', 'OriginalWOA'],
        "optim_paras": [
            {"epoch": 10, "pop_size": 30 },
            {"epoch": 20, "pop_size": 30 },
        ],
        "obj_name": ["F1S"]
    }

    # Hyperparameter tuning with GridSearchCV
    searcher = GridSearchCV(estimator=MhaMlpClassifier(), param_grid=param_grid, cv=3, verbose=2)
    searcher.fit(X_train, y_train)

    # Get best parameters and accuracy
    print("Best Parameters:", searcher.best_params_)
    print("Best Cross-Validation Accuracy:", searcher.best_score_)

    # Evaluate on test set
    best_model = searcher.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)

    print(best_model.network)


def check_MhaMlpRegressor_single_output():
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

    model = MhaMlpRegressor(hidden_layers=(30, 15,), act_names="ELU", dropout_rates=0.2, act_output=None,
                     optim="BaseGA", optim_paras=None, obj_name="MSE", seed=42, verbose=True)
    model.fit(X_train, y_train)
    res = model.score(X_test, y_test)
    print(res)
    print(model.network)


def check_MhaMlpRegressor_single_output_gridsearch():
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
        'optim': ['BaseGA', 'OriginalWOA'],
        "optim_paras": [
            {"epoch": 10, "pop_size": 30},
            {"epoch": 20, "pop_size": 30},
        ],
        "obj_name": ["MSE"]
    }

    # Hyperparameter tuning with GridSearchCV
    searcher = GridSearchCV(estimator=MhaMlpRegressor(), param_grid=param_grid, cv=3,
                            scoring='neg_mean_squared_error', verbose=2)
    searcher.fit(X_train, y_train)

    # Get best parameters and accuracy
    print("Best Parameters:", searcher.best_params_)
    print("Best Cross-Validation Accuracy:", searcher.best_score_)

    # Evaluate on test set
    best_model = searcher.best_estimator_
    # y_pred = best_model.predict(X_test)
    print(best_model.score(X_test, y_test))
    print(best_model.network)


def check_MhaMlpRegressor_multi_output():
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

    model = MhaMlpRegressor(hidden_layers=(30, 15,), act_names="ELU", dropout_rates=0.2, act_output=None,
                     optim="BaseGA", optim_paras=None, obj_name="MSE", seed=42, verbose=True)
    model.fit(X_train, y_train)
    res = model.score(X_test, y_test)
    print(res)
    print(model.network)


def check_MhaMlpRegressor_multi_output_gridsearch():
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
        'optim': ['BaseGA', 'OriginalWOA'],
        "optim_paras": [
            {"epoch": 10, "pop_size": 30},
            {"epoch": 20, "pop_size": 30},
        ],
        "obj_name": ["MSE"]
    }

    # Hyperparameter tuning with GridSearchCV
    searcher = GridSearchCV(estimator=MhaMlpRegressor(), param_grid=param_grid, cv=3,
                            scoring='neg_mean_squared_error', verbose=2)
    searcher.fit(X_train, y_train)

    # Get best parameters and accuracy
    print("Best Parameters:", searcher.best_params_)
    print("Best Cross-Validation Accuracy:", searcher.best_score_)

    # Evaluate on test set
    best_model = searcher.best_estimator_
    # y_pred = best_model.predict(X_test)
    print(best_model.score(X_test, y_test))
    print(best_model.network)


check_MhaMlpClassifier_multi_class()
check_MhaMlpClassifier_multi_class_gridsearch()
check_MhaMlpClassifier_binary_class()
check_MhaMlpClassifier_binary_class_gridsearch()
check_MhaMlpRegressor_single_output()
check_MhaMlpRegressor_single_output_gridsearch()
check_MhaMlpRegressor_multi_output()
check_MhaMlpRegressor_multi_output_gridsearch()
