#!/usr/bin/env python
# Created by "Thieu" at 22:43, 12/01/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Optional, Type, Dict, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim
import banfis.helpers.membership_family as memfa


class ANFIS(nn.Module):
    """Adaptive Neuro-Fuzzy Inference System implementation with multi-output support.

    Args:
        input_dim: Number of input features
        num_rules: Number of fuzzy rules
        output_dim: Number of output dimensions
        membership_class: Class for membership function

    Raises:
        AssertionError: If input parameters are invalid
    """

    SUPPPORT_MEMBERSHIP = {
        "gaussian": "GaussianMembership",
        "triangular": "TriangularMembership",
        "sigmoid": "SigmoidMembership",
        "trapezoidal": "TrapezoidalMembership",
        "pishaped": "PiShapedMembership",
    }

    def __init__(self, input_dim: int, num_rules: int, output_dim: int,
                 membership_class: Union[str, Type[memfa.BaseMembership]]) -> None:
        super(ANFIS, self).__init__()

        # Validate input parameters
        assert input_dim > 0, "input_dim must be positive"
        assert num_rules > 0, "num_rules must be positive"
        assert output_dim > 0, "output_dim must be positive"
        if type(membership_class) is str:
            membership_class = self.SUPPPORT_MEMBERSHIP.get(membership_class)
            if membership_class is None:
                raise ValueError(f"Membership class named {membership_class} is not supported.")
            membership_class = getattr(memfa, membership_class)
        else:
            if not issubclass(membership_class, memfa.BaseMembership):
                raise TypeError(f"if membership_class is a custom, it must inherit from BaseMembership.")

        self.input_dim = input_dim
        self.num_rules = num_rules
        self.output_dim = output_dim

        # Initialize membership functions for each rule
        self.memberships = nn.ModuleList([membership_class(input_dim) for _ in range(num_rules)])

        # Initialize consequent parameters for each rule and each output
        # Shape: (num_rules, input_dim + 1, output_dim)
        self.consequents = nn.Parameter(torch.zeros(num_rules, input_dim + 1, output_dim))

    def get_rule_weights(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate normalized rule firing strengths.

        Args:
            X: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor of normalized rule weights of shape (batch_size, num_rules)
        """
        # Calculate membership values for all rules
        memberships = torch.stack([mf(X) for mf in self.memberships], dim=1)

        # Calculate rule firing strengths
        firing_strengths = torch.prod(memberships, dim=2)

        # Normalize firing strengths
        return firing_strengths / (torch.sum(firing_strengths, dim=1, keepdim=True) + 1e-8)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ANFIS network.

        Args:
            X: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = X.shape[0]

        # Get normalized rule weights (batch_size, num_rules)
        normalized_weights = self.get_rule_weights(X)

        # Prepare input for consequent layer
        X_augmented = torch.cat([X, torch.ones(batch_size, 1, device=X.device)], dim=1)

        # Calculate individual rule outputs
        # (batch_size, input_dim + 1) @ (num_rules, input_dim + 1, output_dim)
        # -> (batch_size, num_rules, output_dim)
        rule_outputs = torch.einsum('bi,rio->bro', X_augmented, self.consequents)

        # Apply normalized weights to rule outputs
        # (batch_size, num_rules, 1) * (batch_size, num_rules, output_dim)
        weighted_outputs = normalized_weights.unsqueeze(-1) * rule_outputs

        # Sum over rules dimension
        # (batch_size, output_dim)
        return torch.sum(weighted_outputs, dim=1)

    def get_rules_info(self, X: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Get information about the fuzzy rules.

        Args:
            X: Optional input tensor to calculate rule activations

        Returns:
            Dictionary containing rule information
        """
        rules_info = {
            'num_rules': self.num_rules,
            'membership_parameters': [mf.get_parameters() for mf in self.memberships],
            'consequent_parameters': self.consequents.data
        }

        if X is not None:
            rules_info['rule_weights'] = self.get_rule_weights(X).detach()

        return rules_info

    def save_model(self, path: str) -> None:
        """
        Save the ANFIS model to a file.

        Args:
            path: Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'num_rules': self.num_rules,
            'output_dim': self.output_dim,
            'membership_class': self.memberships[0].__class__
        }, path)

    @classmethod
    def load_model(cls, path: str) -> 'ANFIS':
        """
        Load an ANFIS model from a file.

        Args:
            path: Path to the saved model

        Returns:
            Loaded ANFIS model
        """
        checkpoint = torch.load(path)
        model = cls(
            checkpoint['input_dim'],
            checkpoint['num_rules'],
            checkpoint['output_dim'],
            checkpoint['membership_class']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


def example_usage():
    """Example usage of the ANFIS network with multiple outputs."""
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Generate dummy data with multiple outputs
    X = torch.rand((200, 5))  # 100 samples, 2 features

    # Số lượng mẫu num_samples = 200 # Tạo dữ liệu ngẫu nhiên cho 5 input đầu vào (X1, X2, X3, X4, X5)
    # X = np.random.rand(num_samples, 5) # Tạo dữ liệu ngẫu nhiên cho 2 output đầu ra (Y1, Y2) Y = np.random.rand(num_samples, 2)

    y = torch.rand((200, 2))
    # Create two target functions
    # y1 = torch.sin(X[:, 0]) + torch.cos(X[:, 1])  # First target
    # y2 = torch.exp(-((X[:, 0]) ** 2 + (X[:, 1]) ** 2))  # Second target
    # y = torch.stack([y1, y2], dim=1)  # Shape: (100, 2)

    # Initialize ANFIS model with 2 outputs
    anfis = ANFIS(input_dim=5, num_rules=5, output_dim=2, membership_class="pishaped")

    #         "gaussian": "GaussianMembership",
    #         "triangular": "TriangularMembership",
    #         "sigmoid": "SigmoidMembership",
    #         "trapezoidal": "TrapezoidalMembership",
    #         "pishaped": "PiShapedMembership",

    # Initialize optimizer and loss function
    optimizer = optim.Adam(anfis.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(500):
        # Forward pass
        predictions = anfis(X)  # Shape: (100, 2)

        # Compute loss
        loss = loss_fn(predictions, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            # Calculate loss for each output separately
            loss_1 = loss_fn(predictions[:, 0], y[:, 0])
            loss_2 = loss_fn(predictions[:, 1], y[:, 1])
            print(f"Epoch {epoch}:")
            print(f"  Total Loss = {loss.item():.4f}")
            print(f"  Output 1 Loss = {loss_1.item():.4f}")
            print(f"  Output 2 Loss = {loss_2.item():.4f}")

    # Test prediction
    test_input = torch.tensor([[0.5, 0.2, 0.1, 0.7, 0.55]], dtype=torch.float32)
    test_prediction = anfis(test_input)
    print("\nTest prediction for input [0.5, 0.2, 0.1, 0.7, 0.55]:")
    print(f"Output 1: {test_prediction[0, 0].item():.4f}")
    print(f"Output 2: {test_prediction[0, 1].item():.4f}")

    # Get and print rules information
    rules_info = anfis.get_rules_info(X)
    print("\nRule weights shape:", rules_info['rule_weights'].shape)
    print("Consequent parameters shape:", rules_info['consequent_parameters'].shape)


if __name__ == "__main__":
    example_usage()
