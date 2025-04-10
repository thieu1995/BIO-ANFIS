#!/usr/bin/env python
# Created by "Thieu" at 23:33, 10/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from banfis.helpers.scaler_util import DataTransformer
from sklearn.model_selection import train_test_split


class OneHotEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        """Fit the encoder to unique categories in X."""
        self.categories_ = np.unique(X)
        return self

    def transform(self, X):
        """Transform X into one-hot encoded format."""
        if self.categories_ is None:
            raise ValueError("The encoder has not been fitted yet.")
        one_hot = np.zeros((X.shape[0], len(self.categories_)), dtype=int)
        for i, val in enumerate(X):
            index = np.where(self.categories_ == val)[0][0]
            one_hot[i, index] = 1
        return one_hot

    def fit_transform(self, X):
        """Fit the encoder to X and transform X."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, one_hot):
        """Convert one-hot encoded format back to original categories."""
        if self.categories_ is None:
            raise ValueError("The encoder has not been fitted yet.")
        if one_hot.shape[1] != len(self.categories_):
            raise ValueError("The shape of the input does not match the number of categories.")
        original = np.array([self.categories_[np.argmax(row)] for row in one_hot])
        return original


class LabelEncoder:
    """
    Encode categorical features as integer labels.
    """

    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    @staticmethod
    def check_y(y):
        y = np.squeeze(np.asarray(y))
        if y.ndim != 1:
            raise ValueError("y label should have shape like 1-D vector.")
        return y

    def fit(self, y):
        """
        Fit label encoder to a given set of labels.

        Parameters:
        -----------
        y : array-like
            Labels to encode.
        """
        y = self.check_y(y)
        self.unique_labels = np.unique(y)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}

    def transform(self, y):
        """
        Transform labels to encoded integer labels.

        Parameters:
        -----------
        y : array-like (1-D vector)
            Labels to encode.

        Returns:
        --------
        encoded_labels : array-like
            Encoded integer labels.
        """
        y = self.check_y(y)
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.label_to_index[label] for label in y])

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform integer labels to original labels.

        Parameters:
        -----------
        y : array-like
            Encoded integer labels.

        Returns:
        --------
        original_labels : array-like
            Original labels.
        """
        y = self.check_y(y)
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.unique_labels[i] if i in self.label_to_index.values() else "unknown" for i in y])


class TimeSeriesDifferencer:

    def __init__(self, interval=1):
        if interval < 1:
            raise ValueError("Interval for differencing must be at least 1.")
        self.interval = interval

    def difference(self, X):
        self.original_data = X.copy()
        return np.array([X[i] - X[i - self.interval] for i in range(self.interval, len(X))])

    def inverse_difference(self, diff_data):
        if self.original_data is None:
            raise ValueError("Original data is required for inversion.")
        return np.array([diff_data[i - self.interval] + self.original_data[i - self.interval] for i in range(self.interval, len(self.original_data))])


class FeatureEngineering:
    def __init__(self):
        """
        Initialize the FeatureEngineering class
        """
        # Check if the threshold is a valid number
        pass

    def create_threshold_binary_features(self, X, threshold):
        """
        Perform feature engineering to add binary indicator columns for values below the threshold.
        Add each new column right after the corresponding original column.

        Args:
        X (numpy.ndarray): The input 2D matrix of shape (n_samples, n_features).
        threshold (float): The threshold value for identifying low values.

        Returns:
        numpy.ndarray: The updated 2D matrix with binary indicator columns.
        """
        # Check if X is a NumPy array
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X should be a NumPy array.")
        # Check if the threshold is a valid number
        if not (isinstance(threshold, int) or isinstance(threshold, float)):
            raise ValueError("Threshold should be a numeric value.")

        # Create a new matrix to hold the original and new columns
        X_new = np.zeros((X.shape[0], X.shape[1] * 2))
        # Iterate over each column in X
        for idx in range(X.shape[1]):
            feature_values = X[:, idx]
            # Create a binary indicator column for values below the threshold
            indicator_column = (feature_values < threshold).astype(int)
            # Add the original column and indicator column to the new matrix
            X_new[:, idx * 2] = feature_values
            X_new[:, idx * 2 + 1] = indicator_column
        return X_new


class Data:
    """
    The structure of our supported Data class

    Parameters
    ----------
    X : np.ndarray
        The features of your data

    y : np.ndarray
        The labels of your data
    """

    SUPPORT = {
        "scaler": list(DataTransformer.SUPPORTED_SCALERS.keys())
    }

    def __init__(self, X=None, y=None, name="Unknown"):
        self.X = X
        self.y = self.check_y(y)
        self.name = name
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    @staticmethod
    def check_y(y):
        if y is None:
            return y
        y = np.squeeze(np.asarray(y))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        return y

    @staticmethod
    def scale(X, scaling_methods=('standard', ), list_dict_paras=None):
        X = np.squeeze(np.asarray(X))
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if X.ndim >= 3:
            raise TypeError(f"Invalid X data type. It should be array-like with shape (n samples, m features)")
        scaler = DataTransformer(scaling_methods=scaling_methods, list_dict_paras=list_dict_paras)
        data = scaler.fit_transform(X)
        return data, scaler

    @staticmethod
    def encode_label(y):
        y = np.squeeze(np.asarray(y))
        if y.ndim != 1:
            raise TypeError(f"Invalid y data type. It should be a vector / array-like with shape (n samples,)")
        scaler = LabelEncoder()
        data = scaler.fit_transform(y)
        return data, scaler

    def split_train_test(self, test_size=0.2, train_size=None,
                         random_state=41, shuffle=True, stratify=None, inplace=True):
        """
        The wrapper of the split_train_test function in scikit-learn library.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                        train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        if not inplace:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def set_train_test(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Function use to set your own X_train, y_train, X_test, y_test in case you don't want to use our split function

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        return self
