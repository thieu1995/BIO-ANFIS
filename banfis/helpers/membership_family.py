#!/usr/bin/env python
# Created by "Thieu" at 22:53, 29/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Optional, Type, Dict, Any
import torch
import torch.nn as nn


class BaseMembership(nn.Module):
    """Base class for membership functions.

    This class defines the interface for all membership functions. Subclasses
    must implement the `forward` method to calculate membership values.
    """

    def __init__(self) -> None:
        super(BaseMembership, self).__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate membership values for the given input tensor.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Tensor of membership values of shape (batch_size,).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve the current parameters of the membership function.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing parameter names and values.
        """
        return {name: param.data for name, param in self.named_parameters()}

    def __str__(self):
        return self.__class__.__name__

    def name(self):
        return self.__class__.__name__


class GaussianMembership(BaseMembership):
    """Gaussian membership function implementation."""

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the Gaussian membership function.

        Args:
            input_dim (int): Number of input features.
        """
        super(GaussianMembership, self).__init__()
        self.centers = nn.Parameter(torch.randn(input_dim))  # Centers
        self.widths = nn.Parameter(torch.abs(torch.randn(input_dim)))  # Widths

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate Gaussian membership values.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Tensor of membership values of shape (batch_size,).
        """
        return torch.exp(-((X - self.centers) ** 2) / (2 * torch.clamp(self.widths, min=1e-8) ** 2))


class TrapezoidalMembership(BaseMembership):
    """Trapezoidal membership function implementation."""

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the Trapezoidal membership function.

        Args:
            input_dim (int): Number of input features.
        """
        super().__init__()
        # Ta khởi tạo theo thứ tự rồi dùng sigmoid để đảm bảo đúng thứ tự a < b < c < d
        self.raw_params = nn.Parameter(torch.rand(4, input_dim))  # [a, b, c, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate Trapezoidal membership values.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Tensor of membership values of shape (batch_size,).
        """
        # Sort theo chiều tăng để đảm bảo thứ tự đúng
        abcd = torch.sort(self.raw_params, dim=0).values
        a, b, c, d = abcd[0], abcd[1], abcd[2], abcd[3]

        ramp_up = torch.clamp((x - a) / (b - a + 1e-6), 0.0, 1.0)
        plateau = torch.ones_like(x)
        ramp_down = torch.clamp((d - x) / (d - c + 1e-6), 0.0, 1.0)

        membership = torch.where(x <= a, 0.0,
                         torch.where(x < b, ramp_up,
                         torch.where(x <= c, plateau,
                         torch.where(x < d, ramp_down, 0.0))))
        return membership


class TriangularMembership(BaseMembership):
    """Triangular membership function implementation."""

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the Triangular membership function.

        Args:
            input_dim (int): Number of input features.
        """
        super(TriangularMembership, self).__init__()
        # Khởi tạo centers trong khoảng [0, 1] vì data của chúng ta cũng nằm trong khoảng này
        self.centers = nn.Parameter(torch.rand(input_dim))
        # Khởi tạo spread với giá trị dương và không quá nhỏ
        self.left_spread = nn.Parameter(torch.ones(input_dim) * 0.5)
        self.right_spread = nn.Parameter(torch.ones(input_dim) * 0.5)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate Triangular membership values.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Tensor of membership values of shape (batch_size,).
        """
        left_spread = torch.abs(self.left_spread) + 1e-4
        right_spread = torch.abs(self.right_spread) + 1e-4

        # Tính toán membership values
        left_side = (X - (self.centers - left_spread)) / left_spread
        right_side = ((self.centers + right_spread) - X) / right_spread

        # Sử dụng torch.relu thay vì torch.maximum để đảm bảo gradient flow tốt hơn
        return torch.minimum(torch.relu(left_side), torch.relu(right_side))


class SigmoidMembership(BaseMembership):
    """Sigmoid membership function implementation."""

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the Sigmoid membership function.

        Args:
            input_dim (int): Number of input features.
        """
        super(SigmoidMembership, self).__init__()
        self.a = nn.Parameter(torch.randn(input_dim))  # Slope
        self.b = nn.Parameter(torch.randn(input_dim))  # Offset

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate Sigmoid membership values.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Tensor of membership values of shape (batch_size,).
        """
        return 1 / (1 + torch.exp(-self.a * (X - self.b)))

