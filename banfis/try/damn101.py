#!/usr/bin/env python
# Created by "Thieu" at 07:53, 11/04/2025 ----------%                                                                               
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

        self.memberships = nn.ModuleList([membership_class(input_dim) for _ in range(num_rules)])
        # Đừng dùng nn.Parameter nữa – để ta cập nhật bằng tay bằng LSE
        self.register_buffer("coeffs", torch.zeros(num_rules, input_dim + 1, output_dim))

    def compute_membership_strengths(self, X):
        memberships = torch.stack([membership(X) for membership in self.memberships], dim=1)
        strengths = torch.prod(memberships, dim=2)
        strengths_sum = torch.sum(strengths, dim=1, keepdim=True)
        normalized_strengths = strengths / (strengths_sum + 1e-8)
        return normalized_strengths

    def compute_rule_outputs(self, X):
        N = X.shape[0]
        X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)  # (N, input_dim + 1)
        # (N, input_dim+1) x (num_rules, input_dim+1, output_dim) → (N, num_rules, output_dim)
        rule_outputs = torch.einsum("ni,rij->nrj", X_aug, self.coeffs)
        return rule_outputs

    def forward(self, X):
        strengths = self.compute_membership_strengths(X)  # (N, num_rules)
        rule_outputs = self.compute_rule_outputs(X)       # (N, num_rules, output_dim)
        output = torch.sum(strengths.unsqueeze(-1) * rule_outputs, dim=1)  # (N, output_dim)
        return output

    def update_consequents_with_LSE(self, X, y_onehot):
        """
        Least Squares Estimation to update consequent coefficients.
        :param X: Input tensor (N, input_dim)
        :param y_onehot: One-hot labels (N, output_dim)
        """
        with torch.no_grad():
            N = X.shape[0]
            X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)  # (N, input_dim + 1)
            strengths = self.compute_membership_strengths(X)  # (N, num_rules)

            design_matrices = strengths.unsqueeze(-1) * X_aug.unsqueeze(1)
            Phi = design_matrices.reshape(N, -1)

            try:
                w_opt = torch.linalg.lstsq(Phi, y_onehot).solution
            except RuntimeError:
                print("LSE failed, skipping update")
                return

            # Reshape to (num_rules, input_dim+1, output_dim)
            self.coeffs.copy_(w_opt.reshape(self.num_rules, self.input_dim + 1, self.output_dim))


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

# Chỉ tối ưu membership parameters
optimizer = optim.Adam([p for name, p in anfis.named_parameters() if "coeffs" not in name], lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# One-hot encode y_train để dùng cho LSE
y_train_onehot = torch.nn.functional.one_hot(y_train, num_classes=10).float()

for epoch in range(500):
    # Step 1: cập nhật hậu đề bằng LSE
    anfis.update_consequents_with_LSE(X_train, y_train_onehot)

    # Step 2: dùng gradient descent cho các membership params
    pred = anfis(X_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")
