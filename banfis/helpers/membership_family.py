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

