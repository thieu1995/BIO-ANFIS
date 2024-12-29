#!/usr/bin/env python
# Created by "Thieu" at 22:33, 28/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%


#
# class SigmoidMembership(BaseMembership):
#     def __init__(self, input_dim):
#         super(SigmoidMembership, self).__init__()
#         self.a = nn.Parameter(torch.randn(input_dim))  # Slope
#         self.b = nn.Parameter(torch.randn(input_dim))  # Offset
#
#     def forward(self, X):
#         return 1 / (1 + torch.exp(-self.a * (X - self.b)))
#
#
# # Trapezoidal Membership Function
# class TrapezoidalMembership(BaseMembership):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.a = nn.Parameter(torch.rand(input_dim))  # Start of the ramp
#         self.b = nn.Parameter(torch.rand(input_dim))  # Start of the plateau
#         self.c = nn.Parameter(torch.rand(input_dim))  # End of the plateau
#         self.d = nn.Parameter(torch.rand(input_dim))  # End of the ramp
#
#     def forward(self, x):
#         ramp_up = (x - self.a) / (self.b - self.a + 1e-6)
#         plateau = torch.ones_like(x)
#         ramp_down = (self.d - x) / (self.d - self.c + 1e-6)
#         membership = torch.min(torch.min(ramp_up, plateau), ramp_down)
#         return torch.clamp(membership, min=0.0)
#
#     def num_trainable_params(self):
#         return 4  # Parameters: a, b, c, d
#
#
# # Pi-Shaped Membership Function
# class PiShapedMembership(BaseMembership):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.a = nn.Parameter(torch.rand(input_dim))  # Start of the ramp
#         self.b = nn.Parameter(torch.rand(input_dim))  # Peak start
#         self.c = nn.Parameter(torch.rand(input_dim))  # Peak end
#         self.d = nn.Parameter(torch.rand(input_dim))  # End of the ramp
#
#     def forward(self, x):
#         ramp_up = (x - self.a) / (self.b - self.a + 1e-6)
#         plateau = torch.ones_like(x)
#         ramp_down = (self.d - x) / (self.d - self.c + 1e-6)
#         rising_edge = torch.clamp(ramp_up, min=0.0, max=1.0)
#         falling_edge = torch.clamp(ramp_down, min=0.0, max=1.0)
#         return torch.min(rising_edge, falling_edge)
#
#     def num_trainable_params(self):
#         return 4  # Parameters: a, b, c, d
#
#


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


class TriangularMembership(BaseMembership):
    def __init__(self, input_dim):
        super(TriangularMembership, self).__init__()
        self.a = nn.Parameter(torch.randn(input_dim))  # Left vertex
        self.b = nn.Parameter(torch.randn(input_dim))  # Peak
        self.c = nn.Parameter(torch.randn(input_dim))  # Right vertex

    def forward(self, X):
        left = (X - self.a) / torch.clamp(self.b - self.a, min=1e-8)
        right = (self.c - X) / torch.clamp(self.c - self.b, min=1e-8)
        return torch.clamp(torch.min(left, right), min=0)



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
        predictions = torch.zeros((N,), device=X.device)
        for r in range(self.num_rules):
            predictions += normalized_strengths[:, r] * (
                torch.matmul(X, self.coeffs[r, :-1]) + self.coeffs[r, -1]
            )

        return predictions, strengths, normalized_strengths

    def forward(self, X):
        N = X.shape[0]

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

        return predictions, strengths, normalized_strengths

    def update_output_coeffs(self, X, y, normalized_strengths):
        """
        Update linear coefficients (self.coeffs) using least squares estimation.
        """
        for r in range(self.num_rules):
            weighted_X = normalized_strengths[:, r].unsqueeze(1) * torch.cat([X, torch.ones((X.size(0), 1), device=X.device)], dim=1)
            A = weighted_X.t().matmul(weighted_X)  # (input_dim+1) x (input_dim+1)
            B = weighted_X.t().matmul(y)  # (input_dim+1) x 1

            # Solve for coefficients using pseudo-inverse
            coeffs = torch.matmul(torch.linalg.pinv(A), B)
            self.coeffs.data[r] = coeffs  # Update coefficients for rule r

    # def update_output_coeffs(self, X, y, normalized_strengths):
    #     """
    #     Update linear coefficients (self.coeffs) using least squares estimation.
    #     """
    #     # Tạo ma trận weighted_X (N, num_rules, input_dim + 1)
    #     weighted_X = normalized_strengths.unsqueeze(2) * torch.cat([X, torch.ones((X.size(0), 1), device=X.device)],
    #                                                                dim=1).unsqueeze(1)
    #
    #     # Mở rộng y thành kích thước (N, num_rules, 1)
    #     y_expanded = y.unsqueeze(1).unsqueeze(2)  # (N, 1, 1) -> (N, num_rules, 1)
    #
    #     # Tính toán ma trận A (num_rules, input_dim + 1, input_dim + 1) và B (num_rules, input_dim + 1, 1)
    #     A = torch.sum(weighted_X.transpose(1, 2) @ weighted_X, dim=0)  # (num_rules, input_dim + 1, input_dim + 1)
    #
    #     # Lấy B với kích thước (num_rules, input_dim + 1, 1)
    #     B = torch.sum(weighted_X.transpose(1, 2) @ (y_expanded * normalized_strengths.unsqueeze(2)),
    #                   dim=0)  # (num_rules, input_dim + 1, 1)
    #
    #     # Giải phương trình LSE bằng pseudo-inverse
    #     coeffs = torch.linalg.pinv(A) @ B  # (num_rules, input_dim + 1, 1)
    #
    #     # Cập nhật hệ số
    #     self.coeffs.data = coeffs.squeeze(2)  # (num_rules, input_dim + 1), bỏ chiều thứ ba


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
        predictions, strengths, normalized_strengths = anfis.forward(X)

        # Update linear parameters (c) using LSE
        anfis.update_output_coeffs(X, y, normalized_strengths)

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



