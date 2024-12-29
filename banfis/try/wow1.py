#!/usr/bin/env python
# Created by "Thieu" at 20:34, 27/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


class ANFIS:
    def __init__(self, input_dim, mf_params, rules):
        """
        Initialize ANFIS model.
        :param input_dim: Number of input variables.
        :param mf_params: List of initial parameters for membership functions [(a, b, c), ...].
        :param rules: List of tuples representing rules (e.g., [(0, 1), (1, 0)]).
        """
        self.input_dim = input_dim
        self.mf_params = np.array(mf_params)  # Parameters for membership functions
        self.rules = rules  # Rules linking input membership functions to output

        # Initialize coefficients for Sugeno-type linear output functions
        self.output_coeffs = np.random.random((len(rules), input_dim + 1))

    def membership_function(self, x, params):
        """
        Gaussian membership function.
        :param x: Input value.
        :param params: Parameters (a, b, c) where b is the mean, c is the standard deviation.
        """
        a, b, c = params
        return 1.0 / (1.0 + ((x - b) / c) ** (2 * a))

    def forward(self, X):
        """
        Forward pass of ANFIS.
        :param X: Input data (N x input_dim).
        :return: Predicted output.
        """
        N, _ = X.shape
        rule_outputs = []
        for rule in self.rules:
            rule_strengths = []
            for i, mf_idx in enumerate(rule):
                mf_value = self.membership_function(X[:, i], self.mf_params[mf_idx])
                rule_strengths.append(mf_value)
            rule_strength = np.prod(rule_strengths, axis=0)  # Rule strength
            rule_outputs.append(rule_strength)
        rule_outputs = np.array(rule_outputs).T  # (N x num_rules)

        # Normalize rule strengths
        normalized_strengths = rule_outputs / np.sum(rule_outputs, axis=1, keepdims=True)

        # Calculate output for each rule
        outputs = []
        for i, coeffs in enumerate(self.output_coeffs):
            rule_output = np.dot(np.hstack((X, np.ones((N, 1)))), coeffs)
            outputs.append(normalized_strengths[:, i] * rule_output)
        return np.sum(outputs, axis=0)

    def forward(self, X):
        """
        Optimized forward pass of ANFIS.
        :param X: Input data (N x input_dim).
        :return: Predicted output.
        """
        N, _ = X.shape
        num_rules = len(self.rules)

        # Compute membership values for all inputs and all membership functions
        mf_values = np.zeros((X.shape[0], len(self.mf_params)))  # (N x num_mfs)
        for i, params in enumerate(self.mf_params):
            mf_values[:, i] = self.membership_function(X[:, i % self.input_dim], params)

        # Compute rule strengths
        rule_strengths = np.ones((N, num_rules))  # (N x num_rules)
        for r, rule in enumerate(self.rules):
            for i, mf_idx in enumerate(rule):
                rule_strengths[:, r] *= mf_values[:, mf_idx]

        # Normalize rule strengths
        normalized_strengths = rule_strengths / np.sum(rule_strengths, axis=1, keepdims=True)

        # Compute rule outputs using matrix operations
        extended_X = np.hstack((X, np.ones((N, 1))))  # Add bias term (N x (input_dim + 1))
        rule_outputs = np.dot(extended_X, self.output_coeffs.T)  # (N x num_rules)
        weighted_outputs = normalized_strengths * rule_outputs  # (N x num_rules)

        # Aggregate outputs
        final_output = np.sum(weighted_outputs, axis=1)  # (N,)
        return final_output

    def train(self, X, y, epochs=100, lr=0.01):
        """
        Train ANFIS model using gradient descent.
        :param X: Training data (N x input_dim).
        :param y: Target output (N,).
        :param epochs: Number of training iterations.
        :param lr: Learning rate.
        """
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            error = y - predictions

            # Update output coefficients (gradient descent)
            for i in range(len(self.rules)):
                for j in range(self.input_dim + 1):
                    gradient = -2 * np.sum(error * self.forward(X))  # Simplified gradient
                    self.output_coeffs[i, j] -= lr * gradient

            print(f"Epoch {epoch + 1}/{epochs}, MSE: {np.mean(error ** 2)}")


# Example usage
if __name__ == "__main__":
    # Example data
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])  # Target output

    # Define membership functions and rules
    mf_params = [(2, 2.0, 1.0), (2, 4.0, 1.0)]  # Parameters for 2 membership functions
    rules = [(0, 0), (1, 1)]  # Example rules

    # Initialize and train ANFIS
    anfis = ANFIS(input_dim=2, mf_params=mf_params, rules=rules)
    anfis.train(X, y, epochs=50, lr=0.01)

    # Predict
    predictions = anfis.forward(X)
    print("Predictions:", predictions)
