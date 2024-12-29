#!/usr/bin/env python
# Created by "Thieu" at 23:32, 28/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn


# Base Membership Class
class BaseMembership(nn.Module):
    def __init__(self, input_dim, num_rules):
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def num_trainable_params(self):
        raise NotImplementedError("Subclasses must implement the num_trainable_params method.")


# Gaussian Membership Function
class GaussianMembership(BaseMembership):
    def __init__(self, input_dim, num_rules):
        super().__init__(input_dim, num_rules)
        self.a = nn.Parameter(torch.rand(num_rules, input_dim))  # Center for each rule
        self.b = nn.Parameter(torch.rand(num_rules, input_dim).abs() + 1e-3)  # Width (must be positive)

    def forward(self, x):
        # Broadcast x to match the shape of a and b
        x_expanded = x.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
        return torch.exp(-((x_expanded - self.a) ** 2) / (2 * self.b ** 2))  # Shape: (batch_size, num_rules, input_dim)

    def num_trainable_params(self):
        return 2 * self.num_rules * self.input_dim  # Parameters: a, b


# Triangular Membership Function
class TriangularMembership(BaseMembership):
    def __init__(self, input_dim, num_rules):
        super().__init__(input_dim, num_rules)
        self.a = nn.Parameter(torch.rand(num_rules, input_dim))  # Left
        self.b = nn.Parameter(torch.rand(num_rules, input_dim))  # Center
        self.c = nn.Parameter(torch.rand(num_rules, input_dim))  # Right

    def forward(self, x):
        x_expanded = x.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
        left = (x_expanded - self.a) / (self.b - self.a + 1e-6)
        right = (self.c - x_expanded) / (self.c - self.b + 1e-6)
        membership = torch.min(left, right)  # Shape: (batch_size, num_rules, input_dim)
        return torch.clamp(membership, min=0.0)

    def num_trainable_params(self):
        return 3 * self.num_rules * self.input_dim  # Parameters: a, b, c


# ANFIS Class
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
#
# class ANFIS(nn.Module):
#     def __init__(self, input_dim, num_rules, membership_class):
#         super().__init__()
#         self.input_dim = input_dim
#         self.num_rules = num_rules
#         self.membership = membership_class(input_dim, num_rules)
#         self.output_coeffs = nn.Parameter(torch.rand(num_rules, input_dim + 1))  # Include bias
#
#     def forward(self, x):
#         # Compute membership values for all rules in parallel
#         membership_values = self.membership(x)  # Shape: (batch_size, num_rules, input_dim)
#         rule_strengths = membership_values.prod(dim=2)  # Product across input_dim, Shape: (batch_size, num_rules)
#
#         # Normalize rule strengths
#         normalized_strengths = rule_strengths / (rule_strengths.sum(dim=1, keepdim=True) + 1e-6)
#
#         # Compute output
#         weighted_outputs = torch.matmul(normalized_strengths, self.output_coeffs[:, :-1])  # No bias
#         bias = self.output_coeffs[:, -1]  # Bias term
#         output = weighted_outputs + bias.unsqueeze(0)  # Broadcast bias to match (batch_size, num_rules)
#         return output, rule_strengths, normalized_strengths
#
#     def forward(self, x):
#         # Compute membership values for all rules in parallel
#         membership_values = self.membership(x)  # Shape: (batch_size, num_rules, input_dim)
#         rule_strengths = membership_values.prod(dim=2)  # Product across input_dim, Shape: (batch_size, num_rules)
#
#         # Normalize rule strengths
#         normalized_strengths = rule_strengths / (rule_strengths.sum(dim=1, keepdim=True) + 1e-6)
#
#         # Compute weighted outputs
#         coeffs = self.output_coeffs[:, :-1]  # Linear coefficients
#         bias = self.output_coeffs[:, -1]  # Bias term
#         weighted_outputs = torch.matmul(normalized_strengths, coeffs) + torch.matmul(normalized_strengths, bias.unsqueeze(1))
#
#         return weighted_outputs, rule_strengths, normalized_strengths

    def update_output_coeffs(self, x, y, normalized_strengths):
        # Compute weighted input matrix for least-squares update
        weighted_x = torch.einsum('br,bi->bri', normalized_strengths, x)  # Shape: (batch_size, num_rules, input_dim)
        weighted_x = torch.cat([weighted_x, normalized_strengths.unsqueeze(2)], dim=2)  # Add bias term
        weighted_x = weighted_x.view(-1, self.input_dim + 1)  # Flatten for least-squares
        weighted_y = (normalized_strengths * y.unsqueeze(1)).view(-1)  # Flatten targets

        # Solve least-squares
        coeffs, _ = torch.lstsq(weighted_y.unsqueeze(1), weighted_x)
        self.output_coeffs.data = coeffs[:self.output_coeffs.numel()].view_as(self.output_coeffs)

    def update_output_coeffs(self, x, y, normalized_strengths):
        # Compute weighted matrix (normalized_strengths.T @ x)
        A = torch.matmul(normalized_strengths.T, x)  # Shape: (num_rules x input_dim)

        # Compute target matrix
        B = torch.matmul(normalized_strengths.T, y)  # Shape: (num_rules x output_dim)

        # Solve using least squares: output_coeffs = (A.T @ A)^-1 @ A.T @ B
        self.output_coeffs.data = torch.linalg.lstsq(A, B).solution

    def num_trainable_params(self):
        return self.membership.num_trainable_params() + self.output_coeffs.numel()


# Test Example
input_dim = 2
num_rules = 3
X = torch.rand(10, input_dim)

anfis = ANFIS(input_dim=input_dim, num_rules=num_rules, membership_class=GaussianMembership)
predictions, rule_strengths, normalized_strengths = anfis.forward(X)
print("Predictions:", predictions)

