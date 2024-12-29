#!/usr/bin/env python
# Created by "Thieu" at 22:00, 29/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.optim as optim


class GaussianMembership(nn.Module):
    def __init__(self, input_dim, num_rules):
        super(GaussianMembership, self).__init__()
        self.a = nn.Parameter(torch.randn(num_rules, input_dim))  # Centers for all rules
        self.b = nn.Parameter(torch.abs(torch.randn(num_rules, input_dim)))  # Widths for all rules

    def forward(self, X):
        """
        Compute membership values for all rules in one pass.
        :param X: Input data (N x input_dim).
        :return: Membership values (N x num_rules x input_dim).
        """
        X = X.unsqueeze(1)  # Shape: (N, 1, input_dim)
        return torch.exp(-((X - self.a) ** 2) / (2 * torch.clamp(self.b, min=1e-8) ** 2))


class ANFIS(nn.Module):
    def __init__(self, input_dim, num_rules):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules

        # Initialize membership functions
        self.memberships = GaussianMembership(input_dim, num_rules)

        # Initialize linear coefficients for each rule
        self.coeffs = nn.Parameter(torch.zeros(num_rules, input_dim + 1))  # Includes bias term

    # def forward(self, X):
    #     N = X.shape[0]
    #
    #     # Calculate membership values for all rules
    #     mu = self.memberships(X)  # Shape: (N, num_rules, input_dim)
    #
    #     # Calculate rule strengths (product of membership values across input_dim)
    #     rule_strengths = torch.prod(mu, dim=2)  # Shape: (N, num_rules)
    #
    #     # Normalize rule strengths
    #     strengths_sum = torch.sum(rule_strengths, dim=1, keepdim=True)
    #     normalized_strengths = rule_strengths / (strengths_sum + 1e-8)  # Shape: (N, num_rules)
    #
    #     # Compute predictions
    #     X_augmented = torch.cat([X, torch.ones((N, 1), device=X.device)], dim=1)  # Add bias term
    #     predictions = torch.matmul(normalized_strengths, torch.matmul(X_augmented, self.coeffs.t()))  # Shape: (N,)
    #
    #     return predictions, rule_strengths, normalized_strengths

    def forward(self, X):
        N = X.shape[0]

        # Calculate membership values for all rules
        mu = self.memberships(X)  # Shape: (N, num_rules, input_dim)

        # Calculate rule strengths (product of membership values across input_dim)
        rule_strengths = torch.prod(mu, dim=2)  # Shape: (N, num_rules)

        # Normalize rule strengths
        strengths_sum = torch.sum(rule_strengths, dim=1, keepdim=True)
        normalized_strengths = rule_strengths / (strengths_sum + 1e-8)  # Shape: (N, num_rules)

        # Compute predictions
        X_augmented = torch.cat([X, torch.ones((N, 1), device=X.device)], dim=1)  # Add bias term
        rule_outputs = torch.matmul(X_augmented, self.coeffs.t())  # Shape: (N, num_rules)
        predictions = torch.sum(normalized_strengths * rule_outputs, dim=1)  # Shape: (N,)

        return predictions, rule_strengths, normalized_strengths

    # def update_output_coeffs(self, X, y, normalized_strengths):
    #     """
    #     Update linear coefficients (self.coeffs) using least squares estimation.
    #     """
    #     N = X.size(0)
    #
    #     # Add bias term to input data
    #     X_augmented = torch.cat([X, torch.ones((N, 1), device=X.device)], dim=1)  # Shape: (N, input_dim+1)
    #
    #     # Calculate weighted input for all rules
    #     weighted_X = normalized_strengths.unsqueeze(2) * X_augmented.unsqueeze(1)  # Shape: (N, num_rules, input_dim+1)
    #
    #     # Batch computation of A and B matrices for all rules
    #     A = torch.einsum('nij,nkj->ijk', weighted_X, weighted_X)  # Shape: (num_rules, input_dim+1, input_dim+1)
    #     B = torch.einsum('nij,nk->ij', weighted_X, y)  # Shape: (num_rules, input_dim+1)
    #
    #     # Solve for coefficients using batched pseudo-inverse
    #     coeffs = torch.matmul(torch.linalg.pinv(A), B.unsqueeze(2)).squeeze(2)  # Shape: (num_rules, input_dim+1)
    #     self.coeffs.data = coeffs
    #
    # def update_output_coeffs(self, X, y, normalized_strengths):
    #     """
    #     Update linear coefficients (self.coeffs) using least squares estimation.
    #     """
    #     N = X.size(0)
    #
    #     # Add bias term to input data
    #     X_augmented = torch.cat([X, torch.ones((N, 1), device=X.device)], dim=1)  # Shape: (N, input_dim+1)
    #
    #     # Calculate weighted input for all rules
    #     weighted_X = normalized_strengths.unsqueeze(2) * X_augmented.unsqueeze(1)  # Shape: (N, num_rules, input_dim+1)
    #
    #     # Batch computation of A and B matrices for all rules
    #     A = torch.einsum('nij,nkj->ijk', weighted_X, weighted_X)  # Shape: (num_rules, input_dim+1, input_dim+1)
    #
    #     # Expand y to match dimensions for einsum
    #     y_expanded = y.unsqueeze(1)  # Shape: (N, 1)
    #
    #     B = torch.einsum('nij,nk->ij', weighted_X, y_expanded)  # Shape: (num_rules, input_dim+1)
    #
    #     # Solve for coefficients using batched pseudo-inverse
    #     coeffs = torch.matmul(torch.linalg.pinv(A), B.unsqueeze(2)).squeeze(2)  # Shape: (num_rules, input_dim+1)
    #     self.coeffs.data = coeffs

    # def update_output_coeffs(self, X, y, normalized_strengths):
    #     """
    #     Update linear coefficients (self.coeffs) using least squares estimation.
    #     """
    #     for r in range(self.num_rules):
    #         weighted_X = normalized_strengths[:, r].unsqueeze(1) * torch.cat([X, torch.ones((X.size(0), 1), device=X.device)], dim=1)
    #         A = weighted_X.t().matmul(weighted_X)  # (input_dim+1) x (input_dim+1)
    #         B = weighted_X.t().matmul(y)  # (input_dim+1) x 1
    #
    #         # Solve for coefficients using pseudo-inverse
    #         coeffs = torch.matmul(torch.linalg.pinv(A), B)
    #         self.coeffs.data[r] = coeffs  # Update coefficients for rule r


if __name__ == "__main__":
    # Generate dummy data
    torch.manual_seed(0)
    X = torch.rand((100, 2))  # 100 samples, 2 features
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1])  # Target values

    # Initialize ANFIS model
    anfis = ANFIS(input_dim=2, num_rules=3)

    # Optimizer for nonlinear parameters (membership params)
    optimizer = optim.Adam(anfis.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(50):
        # Forward pass
        predictions, strengths, normalized_strengths = anfis.forward(X)

        # Update linear parameters (c) using LSE
        # anfis.update_output_coeffs(X, y, normalized_strengths)

        # Compute loss
        loss = loss_fn(predictions, y)

        # Backpropagation for nonlinear parameters (a, b, etc.)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item()}")

    # Final predictions
    final_predictions, _, _ = anfis.forward(X)
    print("Final Loss:", loss_fn(final_predictions, y).item())
