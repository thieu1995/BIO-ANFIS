#!/usr/bin/env python
# Created by "Thieu" at 23:02, 29/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from banfis.helpers.membership_family import GaussianMembership


class ANFIS(nn.Module):
    def __init__(self, input_dim, num_rules, membership_class, num_classes):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.num_classes = num_classes

        # Initialize membership functions
        self.memberships = nn.ModuleList([membership_class(input_dim) for _ in range(num_rules)])

        # Initialize linear coefficients for each rule
        self.coeffs = nn.Parameter(torch.zeros(num_rules, input_dim + 1))  # Includes bias term
        self.output_layer = nn.Linear(num_rules, num_classes)  # Output layer for classification

    def forward(self, X):
        N = X.shape[0]

        # Calculate membership values for all rules (N x num_rules x input_dim)
        memberships = torch.stack([membership(X) for membership in self.memberships], dim=1)

        # Calculate rule strengths by taking the product along the input dimension (dim=2)
        strengths = torch.prod(memberships, dim=2)  # Resulting shape: (N, num_rules)

        # Normalize strengths (N x num_rules)
        strengths_sum = torch.sum(strengths, dim=1, keepdim=True)
        normalized_strengths = strengths / (strengths_sum + 1e-8)

        # Weighted sum of inputs (N x num_rules)
        weighted_inputs = torch.matmul(X, self.coeffs[:, :-1].t()) + self.coeffs[:, -1]  # (N, num_rules)

        # Calculate rule output by multiplying strengths with weighted inputs for each rule
        rule_output = normalized_strengths * weighted_inputs  # (N, num_rules)

        # Pass through output layer to produce class scores
        output = self.output_layer(rule_output)  # (N, num_classes)

        return output


if __name__ == "__main__":
    # Load the Iris dataset from sklearn (multiclass classification)
    data = load_iris()
    X = data.data
    y = data.target

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # Integer labels for classification
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Initialize ANFIS model with Gaussian membership functions
    anfis = ANFIS(input_dim=4, num_rules=5, membership_class=GaussianMembership, num_classes=3)

    # Optimizer and loss function for classification
    optimizer = optim.Adam(anfis.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        # Forward pass
        predictions = anfis.forward(X_train)

        # Compute loss
        loss = loss_fn(predictions, y_train)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

    # Evaluate on the test set
    with torch.no_grad():
        predictions = anfis.forward(X_test)
        _, predicted_classes = torch.max(predictions, 1)
        accuracy = (predicted_classes == y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
