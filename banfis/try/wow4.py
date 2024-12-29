#!/usr/bin/env python
# Created by "Thieu" at 14:51, 28/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.optim as optim


class ANFIS(nn.Module):
    def __init__(self, input_dim, num_rules):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules

        # Initialize rule parameters
        self.rules = nn.ParameterList([
            nn.ParameterDict({
                "a": nn.Parameter(torch.randn(input_dim)),
                "b": nn.Parameter(torch.abs(torch.randn(input_dim))),
                "c": nn.Parameter(torch.zeros(input_dim + 1))  # Linear coefficients + bias
            }) for _ in range(num_rules)
        ])

    def forward(self, X):
        """
        Perform forward propagation to calculate rule strengths, normalized strengths, and predictions.
        :param X: Input data (N x input_dim).
        :return: Predictions, strengths, and normalized strengths.
        """
        N = X.shape[0]
        strengths = []

        # Calculate rule strengths for each rule
        for rule in self.rules:
            mu = []
            for dim in range(self.input_dim):
                # Membership function: Gaussian
                membership = torch.exp(-((X[:, dim] - rule["a"][dim]) ** 2) /
                                       (2 * torch.clamp(rule["b"][dim], min=1e-8) ** 2))
                mu.append(membership)

            # Combine membership functions using product (AND operator)
            rule_strength = torch.prod(torch.stack(mu, dim=1), dim=1)
            strengths.append(rule_strength)

        # Stack rule strengths (N x num_rules)
        strengths = torch.stack(strengths, dim=1)

        # Add epsilon to prevent division by zero
        strengths_sum = torch.sum(strengths, dim=1, keepdim=True)
        normalized_strengths = strengths / (strengths_sum + 1e-8)

        # Predictions (weighted sum of outputs)
        predictions = torch.zeros((N,), device=X.device)
        for r, rule in enumerate(self.rules):
            predictions += normalized_strengths[:, r] * (torch.matmul(X, rule["c"][:-1]) + rule["c"][-1])

        return predictions, strengths, normalized_strengths

    def update_output_coeffs(self, X, y, normalized_strengths):
        """
        Update output coefficients (c) using Least Squares Estimation (LSE).
        :param X: Input data (N x input_dim).
        :param y: Target values (N).
        :param normalized_strengths: Normalized rule strengths (N x num_rules).
        """
        N, _ = X.shape

        # Solve for each rule
        for r, rule in enumerate(self.rules):
            weighted_X = normalized_strengths[:, r].unsqueeze(1) * torch.cat((X, torch.ones((N, 1), device=X.device)), dim=1)
            weighted_y = normalized_strengths[:, r] * y

            # Construct matrices for LSE
            A_T_A = torch.matmul(weighted_X.T, weighted_X)
            A_T_B = torch.matmul(weighted_X.T, weighted_y)

            # Solve using (A^T A)^-1 A^T B
            coeffs = torch.matmul(torch.linalg.pinv(A_T_A), A_T_B)
            rule["c"].data = coeffs


# Example Usage
if __name__ == "__main__":
    # Generate dummy data
    torch.manual_seed(0)
    X = torch.rand((100, 2))  # 100 samples, 2 features
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1])  # Target values

    # Initialize ANFIS model
    anfis = ANFIS(input_dim=2, num_rules=3)

    # Optimizer for nonlinear parameters (a, b)
    optimizer = optim.Adam([param for rule in anfis.rules for param in [rule["a"], rule["b"]]], lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(50):
        # Forward pass
        predictions, strengths, normalized_strengths = anfis.forward(X)

        # Update linear parameters (c) using LSE
        anfis.update_output_coeffs(X, y, normalized_strengths)

        # Compute loss
        loss = loss_fn(predictions, y)

        # Backpropagation for nonlinear parameters (a, b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    # Final predictions
    final_predictions, _, _ = anfis.forward(X)
    print("Final Loss:", loss_fn(final_predictions, y).item())

