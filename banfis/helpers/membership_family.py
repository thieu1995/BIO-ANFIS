#!/usr/bin/env python
# Created by "Thieu" at 22:53, 29/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Optional, Type, Dict, Any
import torch
import torch.nn as nn


class BaseMembership(nn.Module):
    """Base class for membership functions."""

    def __init__(self) -> None:
        super(BaseMembership, self).__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate membership values for given input X.

        Args:
            X: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor of membership values of shape (batch_size,)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get the current parameters of the membership function.

        Returns:
            Dictionary containing parameter names and values
        """
        return {name: param.data for name, param in self.named_parameters()}


class GaussianMembership(BaseMembership):
    """Gaussian membership function implementation."""

    def __init__(self, input_dim: int) -> None:
        """
        Initialize Gaussian membership function.

        Args:
            input_dim: Number of input features
        """
        super(GaussianMembership, self).__init__()
        self.centers = nn.Parameter(torch.randn(input_dim))  # Centers
        self.widths = nn.Parameter(torch.abs(torch.randn(input_dim)))  # Widths

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate Gaussian membership values.

        Args:
            X: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor of membership values of shape (batch_size,)
        """
        return torch.exp(-((X - self.centers) ** 2) / (2 * torch.clamp(self.widths, min=1e-8) ** 2))


# class TriangularMembership(BaseMembership):
#     """Triangular membership function implementation."""
#
#     def __init__(self, input_dim: int) -> None:
#         """
#         Initialize Triangular membership function.
#
#         Args:
#             input_dim: Number of input features
#         """
#         super(TriangularMembership, self).__init__()
#         self.centers = nn.Parameter(torch.randn(input_dim))
#         self.left_spread = nn.Parameter(torch.abs(torch.randn(input_dim)))
#         self.right_spread = nn.Parameter(torch.abs(torch.randn(input_dim)))
#
#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         """
#         Calculate Triangular membership values.
#
#         Args:
#             X: Input tensor of shape (batch_size, input_dim)
#
#         Returns:
#             Tensor of membership values of shape (batch_size,)
#         """
#         left_side = (X - (self.centers - self.left_spread)) / torch.clamp(self.left_spread, min=1e-8)
#         right_side = ((self.centers + self.right_spread) - X) / torch.clamp(self.right_spread, min=1e-8)
#         return torch.maximum(torch.minimum(left_side, right_side), torch.zeros_like(X))


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


class TriangularMembership(BaseMembership):
    def __init__(self, input_dim: int) -> None:
        super(TriangularMembership, self).__init__()
        # Khởi tạo centers trong khoảng [0, 1] vì data của chúng ta cũng nằm trong khoảng này
        self.centers = nn.Parameter(torch.rand(input_dim))
        # Khởi tạo spread với giá trị dương và không quá nhỏ
        self.left_spread = nn.Parameter(torch.ones(input_dim) * 0.5)
        self.right_spread = nn.Parameter(torch.ones(input_dim) * 0.5)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Đảm bảo spread luôn dương và không quá nhỏ
        left_spread = torch.abs(self.left_spread) + 1e-4
        right_spread = torch.abs(self.right_spread) + 1e-4

        # Tính toán membership values
        left_side = (X - (self.centers - left_spread)) / left_spread
        right_side = ((self.centers + right_spread) - X) / right_spread

        # Sử dụng torch.relu thay vì torch.maximum để đảm bảo gradient flow tốt hơn
        membership = torch.minimum(
            torch.relu(left_side),
            torch.relu(right_side)
        )

        return membership

#
# class GaussianMembership(BaseMembership):
#     def __init__(self, input_dim):
#         super(GaussianMembership, self).__init__()
#         self.a = nn.Parameter(torch.randn(input_dim))  # Centers
#         self.b = nn.Parameter(torch.abs(torch.randn(input_dim)))  # Widths
#
#     def forward(self, X):
#         return torch.exp(-((X - self.a) ** 2) / (2 * torch.clamp(self.b, min=1e-8) ** 2))


class SigmoidMembership(BaseMembership):
    def __init__(self, input_dim):
        super(SigmoidMembership, self).__init__()
        self.a = nn.Parameter(torch.randn(input_dim))  # Slope
        self.b = nn.Parameter(torch.randn(input_dim))  # Offset

    def forward(self, X):
        return 1 / (1 + torch.exp(-self.a * (X - self.b)))


# Trapezoidal Membership Function
class TrapezoidalMembership(BaseMembership):
    def __init__(self, input_dim):
        super().__init__()
        self.a = nn.Parameter(torch.rand(input_dim))  # Start of the ramp
        self.b = nn.Parameter(torch.rand(input_dim))  # Start of the plateau
        self.c = nn.Parameter(torch.rand(input_dim))  # End of the plateau
        self.d = nn.Parameter(torch.rand(input_dim))  # End of the ramp

    def forward(self, x):
        ramp_up = (x - self.a) / (self.b - self.a + 1e-6)
        plateau = torch.ones_like(x)
        ramp_down = (self.d - x) / (self.d - self.c + 1e-6)
        membership = torch.min(torch.min(ramp_up, plateau), ramp_down)
        return torch.clamp(membership, min=0.0)

    def num_trainable_params(self):
        return 4  # Parameters: a, b, c, d


# Pi-Shaped Membership Function
class PiShapedMembership(BaseMembership):
    def __init__(self, input_dim):
        super().__init__()
        self.a = nn.Parameter(torch.rand(input_dim))  # Start of the ramp
        self.b = nn.Parameter(torch.rand(input_dim))  # Peak start
        self.c = nn.Parameter(torch.rand(input_dim))  # Peak end
        self.d = nn.Parameter(torch.rand(input_dim))  # End of the ramp

    def forward(self, x):
        ramp_up = (x - self.a) / (self.b - self.a + 1e-6)
        plateau = torch.ones_like(x)
        ramp_down = (self.d - x) / (self.d - self.c + 1e-6)
        rising_edge = torch.clamp(ramp_up, min=0.0, max=1.0)
        falling_edge = torch.clamp(ramp_down, min=0.0, max=1.0)
        return torch.min(rising_edge, falling_edge)

    def num_trainable_params(self):
        return 4  # Parameters: a, b, c, d
