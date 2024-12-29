#!/usr/bin/env python
# Created by "Thieu" at 20:36, 27/12/2024 ----------%
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
        Optimized forward pass of ANFIS.
        :param X: Input data (N x input_dim).
        :return: Predicted output, rule strengths, normalized rule strengths.
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
        return final_output, rule_strengths, normalized_strengths

    def train(self, X, y, epochs=100, lr=0.01):
        """
        Train ANFIS model using gradient descent for membership functions and LSE for output coefficients.
        :param X: Training data (N x input_dim).
        :param y: Target output (N,).
        :param epochs: Number of training iterations.
        :param lr: Learning rate.
        """
        for epoch in range(epochs):
            # Forward pass
            predictions, rule_strengths, normalized_strengths = self.forward(X)
            error = y - predictions

            # Update output coefficients (LSE approach)
            extended_X = np.hstack((X, np.ones((X.shape[0], 1))))  # Add bias term
            for i in range(len(self.rules)):
                weighted_X = extended_X * normalized_strengths[:, i:i + 1]  # Weight inputs by rule strengths
                self.output_coeffs[i] += lr * np.dot(weighted_X.T, error) / len(X)

            # Update membership function parameters (Gradient Descent)
            for i, params in enumerate(self.mf_params):
                a, b, c = params
                for n in range(X.shape[0]):
                    x_n = X[n, i % self.input_dim]
                    mu = self.membership_function(x_n, params)
                    d_mu_db = mu * 2 * a * (x_n - b) / (c ** 2)
                    d_mu_dc = mu * 2 * a * (x_n - b) ** 2 / (c ** 3)

                    # Chain rule to propagate error
                    gradient_b = -2 * error[n] * d_mu_db
                    gradient_c = -2 * error[n] * d_mu_dc

                    # Update parameters
                    self.mf_params[i, 1] -= lr * gradient_b  # Update b
                    self.mf_params[i, 2] -= lr * gradient_c  # Update c

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
    predictions, _, _ = anfis.forward(X)
    print("Predictions:", predictions)
