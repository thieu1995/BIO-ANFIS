#!/usr/bin/env python
# Created by "Thieu" at 22:29, 29/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn

class GaussianMembershipFunction(nn.Module):
    def __init__(self, input_dim):
        super(GaussianMembershipFunction, self).__init__()
        self.centers = nn.Parameter(torch.randn(input_dim))  # centers for each input dimension
        self.sigmas = nn.Parameter(torch.ones(input_dim))  # sigmas for each input dimension

    def forward(self, X):
        # Gaussian membership function for each input dimension
        diff = X.unsqueeze(1) - self.centers
        exp_term = torch.exp(-0.5 * (diff / self.sigmas) ** 2)
        return exp_term


class ANFIS(nn.Module):
    def __init__(self, input_dim, num_rules, output_dim, membership_class):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.output_dim = output_dim

        # Initialize membership functions
        self.memberships = nn.ModuleList([membership_class(input_dim) for _ in range(num_rules)])

        # Initialize linear coefficients for each rule (coefficients for multi-output)
        self.coeffs = nn.Parameter(torch.zeros(num_rules, input_dim + 1, output_dim))  # Includes bias term

    def forward(self, X):
        N = X.shape[0]
        strengths = []

        # Calculate rule strengths
        for membership in self.memberships:
            mu = membership(X)  # Membership output (N x input_dim)
            rule_strength = torch.prod(mu, dim=1)  # Combine memberships using product
            strengths.append(rule_strength)

        # Stack rule strengths (N x num_rules)
        strengths = torch.stack(strengths, dim=1)

        # Normalize strengths
        strengths_sum = torch.sum(strengths, dim=1, keepdim=True)
        normalized_strengths = strengths / (strengths_sum + 1e-8)

        # Predictions (weighted sum of outputs)
        predictions = torch.zeros((N, self.output_dim), device=X.device)
        for r in range(self.num_rules):
            rule_output = torch.matmul(X, self.coeffs[r, :-1]) + self.coeffs[r, -1]
            predictions += normalized_strengths[:, r].unsqueeze(1) * rule_output

        return predictions, strengths, normalized_strengths

    # def update_output_coeffs(self, X, y, normalized_strengths):
    #     """
    #     Update linear coefficients (self.coeffs) using least squares estimation for multi-output regression.
    #     """
    #     for r in range(self.num_rules):
    #         weighted_X = normalized_strengths[:, r].unsqueeze(1) * torch.cat([X, torch.ones((X.size(0), 1), device=X.device)], dim=1)
    #         A = weighted_X.t().matmul(weighted_X)  # (input_dim+1) x (input_dim+1)
    #         B = weighted_X.t().matmul(y)  # (input_dim+1) x output_dim
    #
    #         # Solve for coefficients using pseudo-inverse
    #         coeffs = torch.matmul(torch.linalg.pinv(A), B)
    #         self.coeffs.data[r] = coeffs  # Update coefficients for rule r

# Assume multi-output regression with MSE loss
def train(model, X_train, y_train, optimizer, num_epochs=100):
    criterion = nn.MSELoss()  # For multi-output regression
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        predictions, _, normalized_strengths = model(X_train)

        # Compute loss
        loss = criterion(predictions, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

# Assume multi-class classification with CrossEntropyLoss
def train_classification(model, X_train, y_train, optimizer, num_epochs=100):
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        predictions, _, normalized_strengths = model(X_train)

        # Compute loss
        loss = criterion(predictions, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")


# # Example: multi-output regression
# input_dim = 2  # Example input dimension
# num_rules = 3  # Number of rules in the network
# output_dim = 2  # Number of output dimensions for multi-output regression
#
# # Example dataset
# X_train = torch.randn(100, input_dim)  # 100 samples, 2 features
# y_train = torch.randn(100, output_dim)  # 100 samples, 2 outputs
#
# # Create ANFIS model
# model = ANFIS(input_dim, num_rules, output_dim, GaussianMembershipFunction)
#
# # Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# # Training the model
# train(model, X_train, y_train, optimizer)


# Example: multi-class classification
input_dim = 2  # Example input dimension
num_rules = 3  # Number of rules in the network
num_classes = 3  # Number of classes in multi-class classification

# Example dataset
X_train = torch.randn(100, input_dim)  # 100 samples, 2 features
y_train = torch.randint(0, num_classes, (100,))  # 100 samples, class labels in the range [0, num_classes)

# Create ANFIS model
model = ANFIS(input_dim, num_rules, num_classes, GaussianMembershipFunction)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
train_classification(model, X_train, y_train, optimizer)
