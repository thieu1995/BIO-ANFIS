#!/usr/bin/env python
# Created by "Thieu" at 15:27, 29/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Membership base class
class BaseMembership(nn.Module):
    def __init__(self, input_dim, num_rules):
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules

    def forward(self, x):
        raise NotImplementedError("This method should be implemented by subclasses!")

# Gaussian Membership Function
class GaussianMembership(BaseMembership):
    def __init__(self, input_dim, num_rules):
        super().__init__(input_dim, num_rules)
        self.mean = nn.Parameter(torch.randn(num_rules, input_dim))
        self.std = nn.Parameter(torch.ones(num_rules, input_dim))

    def forward(self, x):
        return torch.exp(-0.5 * ((x.unsqueeze(1) - self.mean) / (self.std + 1e-6)) ** 2)

# ANFIS class
class ANFIS(nn.Module):
    def __init__(self, input_dim, num_rules, output_dim, membership_class):
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.output_dim = output_dim

        # Membership functions
        self.membership = membership_class(input_dim, num_rules)

        # Output coefficients (shape: num_rules x output_dim)
        self.output_coeffs = nn.Parameter(torch.randn(num_rules, output_dim))

    def forward(self, x):
        # Compute membership values for all rules in parallel
        membership_values = self.membership(x)  # Shape: (batch_size, num_rules, input_dim)
        rule_strengths = membership_values.prod(dim=2)  # Product across input_dim, Shape: (batch_size, num_rules)

        # Normalize rule strengths
        normalized_strengths = rule_strengths / (rule_strengths.sum(dim=1, keepdim=True) + 1e-6)

        # Compute output (shape: batch_size x output_dim)
        output = torch.matmul(normalized_strengths, self.output_coeffs)

        return output, rule_strengths, normalized_strengths

    def update_output_coeffs(self, x, y, normalized_strengths):
        A = normalized_strengths.T  # Transpose normalized strengths (num_rules x batch_size)
        B = y  # Labels (batch_size x output_dim)

        # Solve using least squares: output_coeffs = (A.T @ A)^-1 @ A.T @ B
        pseudo_inverse = torch.linalg.pinv(A.T @ A)
        self.output_coeffs.data = torch.matmul(pseudo_inverse, A.T @ B)

# Generate synthetic regression data
def create_dataset(n_samples=1000, n_features=4, noise=0.1):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    y = y.reshape(-1, 1)  # Ensure y has shape (n_samples, 1)
    return X, y

# Main function
def main():
    # Dataset preparation
    X, y = create_dataset(n_samples=500, n_features=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train, X_test = scaler_X.fit_transform(X_train), scaler_X.transform(X_test)
    y_train, y_test = scaler_y.fit_transform(y_train), scaler_y.transform(y_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Model initialization
    input_dim = X_train.shape[1]
    num_rules = 5
    output_dim = y_train.shape[1]
    anfis = ANFIS(input_dim=input_dim, num_rules=num_rules, output_dim=output_dim, membership_class=GaussianMembership)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(anfis.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions, _, normalized_strengths = anfis.forward(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        # Update output coefficients
        anfis.update_output_coeffs(X_train, y_train, normalized_strengths)


        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Testing
    predictions, _, _ = anfis.forward(X_test)
    test_loss = criterion(predictions, y_test).item()
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
