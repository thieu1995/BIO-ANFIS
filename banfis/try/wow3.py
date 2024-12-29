#!/usr/bin/env python
# Created by "Thieu" at 14:27, 28/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import numpy as np


class ANFIS(nn.Module):
    def __init__(self, input_dim, mf_params, rules):
        """
        Initialize ANFIS model.
        :param input_dim: Number of input variables.
        :param mf_params: Initial parameters for membership functions [(a, b, c), ...].
        :param rules: List of tuples representing rules (e.g., [(0, 1), (1, 0)]).
        """
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.rules = rules  # Rules linking input membership functions to output

        # Membership function parameters (a, b, c)
        self.mf_params = nn.Parameter(torch.tensor(mf_params, dtype=torch.float32))

        # Initialize coefficients for Sugeno-type linear output functions
        self.output_coeffs = torch.randn(len(rules), input_dim + 1, requires_grad=False)

    def membership_function(self, x, params):
        """
        Gaussian membership function.
        :param x: Input value (Tensor).
        :param params: Parameters (a, b, c).
        """
        a, b, c = params
        return 1.0 / (1.0 + ((x - b) / c) ** (2 * a))

    def forward(self, X):
        """
        Perform forward propagation to calculate rule strengths and normalized strengths.
        :param X: Input data (N x input_dim).
        :return: Normalized strengths (N x num_rules).
        """
        N = X.shape[0]
        strengths = []

        # Calculate rule strengths for each rule
        for rule in self.rules:
            mu = []
            for dim in range(self.input_dim):
                membership = torch.exp(-((X[:, dim] - rule["a"][dim]) ** 2) / (2 * rule["b"][dim] ** 2))
                mu.append(membership)

            # Combine membership functions using product (AND operator)
            rule_strength = torch.prod(torch.stack(mu, dim=1), dim=1)
            strengths.append(rule_strength)

        # Stack rule strengths (N x num_rules)
        strengths = torch.stack(strengths, dim=1)

        # Add epsilon to prevent division by zero
        strengths_sum = torch.sum(strengths, dim=1, keepdim=True)
        normalized_strengths = strengths / (strengths_sum + 1e-8)

        # Check for non-finite values
        if not torch.isfinite(strengths).all():
            raise ValueError("Non-finite values in strengths")
        if not torch.isfinite(normalized_strengths).all():
            raise ValueError("Non-finite values in normalized_strengths")

        return normalized_strengths

    def update_output_coeffs(self, X, y, normalized_strengths):
        """
        Update the output coefficients using Least Squares Estimation (LSE).
        :param X: Input data (N x input_dim).
        :param y: Target output (N,).
        :param normalized_strengths: Normalized rule strengths (N x num_rules).
        """
        with torch.no_grad():
            N, _ = X.shape
            extended_X = torch.cat((X, torch.ones((N, 1), dtype=torch.float32)), dim=1)  # Add bias term

            for r in range(len(self.rules)):
                # Weighted inputs
                weighted_X = extended_X * normalized_strengths[:, r:r + 1]
                weighted_y = y * normalized_strengths[:, r]

                # Check for NaN/Inf in inputs
                if not torch.isfinite(weighted_X).all():
                    raise ValueError(f"Non-finite values in weighted_X for rule {r}")
                if not torch.isfinite(weighted_y).all():
                    raise ValueError(f"Non-finite values in weighted_y for rule {r}")

                # Compute (A^T A) and (A^T B)
                A_T_A = torch.matmul(weighted_X.T, weighted_X)
                A_T_B = torch.matmul(weighted_X.T, weighted_y)

                # Check for NaN/Inf in computed matrices
                if not torch.isfinite(A_T_A).all():
                    raise ValueError(f"Non-finite values in A_T_A for rule {r}")
                if not torch.isfinite(A_T_B).all():
                    raise ValueError(f"Non-finite values in A_T_B for rule {r}")

                # Solve (A^T A)^(-1) A^T B
                coeffs = torch.matmul(torch.linalg.pinv(A_T_A), A_T_B)
                self.output_coeffs[r] = coeffs


# Example usage
if __name__ == "__main__":
    # Example data
    X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=torch.float32)
    y = torch.tensor([3.0, 5.0, 7.0, 9.0], dtype=torch.float32)

    # Define membership functions and rules
    mf_params = [[2.0, 2.0, 1.0], [2.0, 4.0, 1.0]]  # Parameters for 2 membership functions
    rules = [(0, 0), (1, 1)]  # Example rules

    # Initialize ANFIS model
    anfis = ANFIS(input_dim=2, mf_params=mf_params, rules=rules)

    # Optimizer for membership function parameters
    optimizer = torch.optim.Adam([anfis.mf_params], lr=0.01)

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        predictions, _, normalized_strengths = anfis.forward(X)
        loss = torch.mean((y - predictions) ** 2)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update output coefficients with LSE
        anfis.update_output_coeffs(X, y, normalized_strengths)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Predict
    predictions, _, _ = anfis.forward(X)
    print("Predictions:", predictions)
