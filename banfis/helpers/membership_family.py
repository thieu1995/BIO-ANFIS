#!/usr/bin/env python
# Created by "Thieu" at 22:53, 29/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn


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
