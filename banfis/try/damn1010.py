#!/usr/bin/env python
# Created by "Thieu" at 08:06, 11/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BaseMembership(nn.Module):
    def __init__(self):
        super(BaseMembership, self).__init__()

    def forward(self, X):
        raise NotImplementedError("Subclasses must implement the forward method.")

class GaussianMembership(BaseMembership):
    def __init__(self, input_dim):
        super(GaussianMembership, self).__init__()
        self.a = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.abs(torch.randn(input_dim)))

    def forward(self, X):
        return torch.exp(-((X - self.a) ** 2) / (2 * torch.clamp(self.b, min=1e-8) ** 2))

class ANFIS(nn.Module):
    def __init__(self, input_dim, num_rules, output_dim, membership_class):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.output_dim = output_dim
        self.memberships = nn.ModuleList([membership_class(input_dim) for _ in range(num_rules)])
        self.coeffs = nn.Parameter(torch.zeros(num_rules, input_dim + 1, output_dim))

    def compute_membership_strengths(self, X):
        memberships = torch.stack([m(X) for m in self.memberships], dim=1)
        # log_strengths = torch.sum(torch.log(memberships + 1e-6), dim=2)
        # strengths = torch.exp(log_strengths)
        strengths = torch.mean(memberships, dim=2)
        # print(strengths)
        normalized_strengths = strengths / (torch.sum(strengths, dim=1, keepdim=True) + 1e-8)
        return normalized_strengths

    def forward(self, X):
        normalized_strengths = self.compute_membership_strengths(X)
        weighted_inputs = torch.einsum("ni,rij->nrj", X, self.coeffs[:, :-1, :]) + self.coeffs[:, -1, :]
        output = torch.sum(normalized_strengths.unsqueeze(-1) * weighted_inputs, dim=1)
        return output

    def update_consequents_with_LSE(self, X, y):
        with torch.no_grad():
            N = X.shape[0]
            ones = torch.ones(N, 1)
            X_bias = torch.cat([X, ones], dim=1)
            normalized_strengths = self.compute_membership_strengths(X)

            # Bước 3: Vectorized: Nhân strengths với X_bias
            X_expanded = X_bias.unsqueeze(1)  # (N, 1, input_dim + 1)
            strengths_expanded = normalized_strengths.unsqueeze(2)  # (N, num_rules, 1)
            F = strengths_expanded * X_expanded  # (N, num_rules, input_dim + 1)
            F = F.reshape(N, -1)  # (N, num_rules * (input_dim + 1))

            # F = []
            # for r in range(self.num_rules):
            #     wr = normalized_strengths[:, r].unsqueeze(1)
            #     Xr = wr * X_bias
            #     F.append(Xr)
            # F = torch.cat(F, dim=1)
            try:
                # coeffs_flat, _ = torch.linalg.lstsq(y, F)
                coeffs_flat = torch.linalg.lstsq(F, y, driver='gelsd').solution
                print("fuck")
            except:
                coeffs_flat = torch.linalg.pinv(F) @ y
            coeffs = coeffs_flat[:F.shape[1]].reshape(self.num_rules, self.input_dim + 1, self.output_dim)
            self.coeffs.data.copy_(coeffs)

# Load and preprocess data
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Initialize model
input_dim = 64
num_rules = 20
output_dim = 10
anfis = ANFIS(input_dim, num_rules, output_dim, GaussianMembership)
loss_fn = nn.CrossEntropyLoss()

# # Select training mode
# training_mode = "hybrid"  # options: "full_gd", "hybrid"
#
# if training_mode == "full_gd":
#     optimizer = optim.Adam(anfis.parameters(), lr=0.001)
#     for epoch in range(100):
#         anfis.train()
#         pred = anfis(X_train)
#         loss = loss_fn(pred, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"[Full GD] Epoch {epoch:03d}: Loss = {loss.item():.6f}")
#
# elif training_mode == "hybrid":
#     optimizer = optim.Adam([p for name, p in anfis.named_parameters() if 'coeffs' not in name], lr=0.1)
#     # loss_fn = nn.MSELoss()
#
#     for epoch in range(100):
#         anfis.train()
#         # Step 1: update consequent parameters using LSE
#         # y_onehot = torch.nn.functional.one_hot(y_train, num_classes=output_dim).float()
#         # pred = anfis(X_train)
#         # loss = loss_fn(pred, y_onehot)
#
#         y_onehot = torch.nn.functional.one_hot(y_train, num_classes=output_dim).float()
#         anfis.update_consequents_with_LSE(X_train.detach(), y_onehot.detach())
#
#         # Step 2: update MF parameters with GD
#         pred = anfis(X_train)
#         loss = loss_fn(pred, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"[Hybrid] Epoch {epoch:03d}: Loss = {loss.item():.4f}")
#
# else:
#     raise ValueError("Invalid training mode. Use 'full_gd' or 'hybrid'")


# ---- Training Parameters ----
training_mode = "hybrid"  # or "full_gd"
num_epochs = 100
learning_rate = 0.01
use_mse_in_hybrid = True  # Set False to try CrossEntropyLoss instead

# ---- Optimizer Setup ----
if training_mode == "full_gd":
    optimizer = optim.Adam(anfis.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

elif training_mode == "hybrid":
    # Freeze consequent parameters during GD
    optimizer = optim.Adam([p for name, p in anfis.named_parameters() if 'coeffs' not in name], lr=learning_rate)
    if use_mse_in_hybrid:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

# ---- Training Loop ----
for epoch in range(num_epochs):
    anfis.train()

    if training_mode == "hybrid":
        # Convert labels to one-hot
        y_onehot = F.one_hot(y_train, num_classes=anfis.output_dim).float()

        # Step 1: update consequent parameters using LSE
        anfis.update_consequents_with_LSE(X_train.detach(), y_onehot.detach())

        # Step 2: forward pass
        pred = anfis(X_train)

        # Step 3: compute loss
        if use_mse_in_hybrid:
            loss = loss_fn(pred, y_onehot)
        else:
            loss = loss_fn(pred, y_train)

    elif training_mode == "full_gd":
        pred = anfis(X_train)
        loss = loss_fn(pred, y_train)

    # Step 4: update premise (membership) parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} | Loss = {loss.item():.6f}")


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
