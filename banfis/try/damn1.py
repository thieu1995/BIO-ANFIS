#!/usr/bin/env python
# Created by "Thieu" at 15:21, 11/01/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.optim as optim


class BaseMembership(nn.Module):
    def __init__(self):
        super(BaseMembership, self).__init__()

    def forward(self, X):
        """
        Calculate membership values for given input X.
        :param X: Input data (N x input_dim).
        :param params: Parameters for the membership function.
        :return: Membership values (N,).
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class GaussianMembership(BaseMembership):
    def __init__(self, input_dim):
        super(GaussianMembership, self).__init__()
        self.a = nn.Parameter(torch.randn(input_dim))  # Centers
        self.b = nn.Parameter(torch.abs(torch.randn(input_dim)))  # Widths

    def forward(self, X):
        return torch.exp(-((X - self.a) ** 2) / (2 * torch.clamp(self.b, min=1e-8) ** 2))


class ANFIS(nn.Module):
    def __init__(self, input_dim, num_rules, membership_class):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules

        # Initialize membership functions
        self.memberships = nn.ModuleList([membership_class(input_dim) for _ in range(num_rules)])

        # Initialize linear coefficients for each rule
        self.coeffs = nn.Parameter(torch.zeros(num_rules, input_dim + 1))  # Includes bias term

    def forward(self, X):
        # Calculate membership values for all rules (N x num_rules x input_dim)
        memberships = torch.stack([membership(X) for membership in self.memberships], dim=1)

        # Calculate rule strengths by taking the product along the input dimension (dim=2)
        strengths = torch.prod(memberships, dim=2)  # Resulting shape: (N, num_rules)

        # Normalize strengths (N x num_rules)
        strengths_sum = torch.sum(strengths, dim=1, keepdim=True)
        normalized_strengths = strengths / (strengths_sum + 1e-8)

        # Predictions (weighted sum of outputs)
        weighted_inputs = torch.matmul(X, self.coeffs[:, :-1].t()) + self.coeffs[:, -1]  # (N, num_rules)
        predictions = torch.sum(normalized_strengths * weighted_inputs, dim=1)

        return predictions


if __name__ == "__main__":
    # Generate dummy data
    torch.manual_seed(0)
    X = torch.rand((100, 2))  # 100 samples, 2 features
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1])  # Target values

    # Initialize ANFIS model with Gaussian membership
    anfis = ANFIS(input_dim=2, num_rules=3, membership_class=GaussianMembership)

    # Optimizer for nonlinear parameters (membership params)
    optimizer = optim.Adam(anfis.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(50):
        # Forward pass
        predictions = anfis.forward(X)

        # Compute loss
        loss = loss_fn(predictions, y)

        # Backpropagation for nonlinear parameters (a, b, etc.)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")

    # Final predictions
    final_predictions = anfis.forward(X)
    print("Final Loss:", loss_fn(final_predictions, y).item())

