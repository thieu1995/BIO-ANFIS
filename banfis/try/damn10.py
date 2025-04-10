#!/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    def __init__(self, input_dim, num_rules, output_dim, membership_class):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.output_dim = output_dim

        # Initialize membership functions
        self.memberships = nn.ModuleList([membership_class(input_dim) for _ in range(num_rules)])

        # Initialize linear coefficients for each rule
        # self.coeffs = nn.Parameter(torch.zeros(num_rules, input_dim + 1))  # Includes bias term (num_rules, input_dim + 1)
        self.coeffs = nn.Parameter(torch.zeros(num_rules, input_dim + 1, output_dim))  # (num_rules, input_dim+1, output_dim)

    def forward(self, X):
        # Calculate membership values for all rules (N x num_rules x input_dim)
        memberships = torch.stack([membership(X) for membership in self.memberships], dim=1)

        # Fix 1: Dùng log-product thay vì product để tránh underflow
        # log_strengths = torch.sum(torch.log(memberships + 1e-6), dim=2)  # log-product
        # strengths = torch.exp(log_strengths)

        # Fix 2: Dùng mean thay vì product (đơn giản hơn, dễ hội tụ)
        strengths = torch.mean(memberships, dim=2)

        # Original
        # Calculate rule strengths by taking the product along the input dimension (dim=2)
        # strengths = torch.prod(memberships, dim=2)  # Resulting shape: (N, num_rules)

        # Normalize strengths (N x num_rules)
        strengths_sum = torch.sum(strengths, dim=1, keepdim=True)
        normalized_strengths = strengths / (strengths_sum + 1e-8)

        # # Predictions (weighted sum of outputs)
        # weighted_inputs = torch.matmul(X, self.coeffs[:, :-1].t()) + self.coeffs[:, -1]  # (N, num_rules)
        # predictions = torch.sum(normalized_strengths * weighted_inputs, dim=1)  # (N,)
        # return predictions

        # Prepare input for consequent layer
        # (N, num_rules, input_dim)
        weighted_inputs = torch.einsum("ni,rij->nrj", X, self.coeffs[:, :-1, :]) + self.coeffs[:, -1,:]  # (N, num_rules, output_dim)
        # Multiply with normalized strengths (broadcasting): (N, num_rules, 1) * (N, num_rules, output_dim)
        output = torch.sum(normalized_strengths.unsqueeze(-1) * weighted_inputs, dim=1)  # (N, output_dim)
        return output  # logits, to be passed to CrossEntropyLoss


# Load and preprocess data
digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Initialize model
anfis = ANFIS(input_dim=64, num_rules=20, output_dim=10, membership_class=GaussianMembership)
optimizer = optim.Adam(anfis.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(500):
    pred = anfis(X_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    # for name, param in anfis.named_parameters():
    #     if param.grad is not None:
    #         print(f"{name}: grad norm = {param.grad.norm().item():.4f}")

    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")

# Accuracy
with torch.no_grad():
    pred_test = anfis(X_test)
    predicted_classes = torch.argmax(pred_test, dim=1)
    acc = (predicted_classes == y_test).float().mean()
    print("Test Accuracy:", acc.item())


    print("Model Parameters:")
    for name, param in anfis.named_parameters():
        print(f"Name: {name}")
        print(f"Shape: {param.shape}")
        print(f"Values:\n{param.data}\n")

    total_params = sum(p.numel() for p in anfis.parameters())
    print(f"Total number of parameters: {total_params}")

