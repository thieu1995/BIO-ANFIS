#!/usr/bin/env python
# Created by "Thieu" at 21:55, 29/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn

class ANFIS(nn.Module):
    def __init__(self, input_dim, num_rules, num_outputs):
        """
        Simplified ANFIS model focusing only on num_rules.

        Args:
            input_dim (int): Number of input features.
            num_rules (int): Number of fuzzy rules.
            num_outputs (int): Number of output dimensions.
        """
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.num_outputs = num_outputs

        # Parameters for rules: each rule has weights and bias
        self.rule_weights = nn.Parameter(torch.randn(num_rules, input_dim))  # Weights for each rule
        self.rule_biases = nn.Parameter(torch.randn(num_rules))             # Biases for each rule

        # Parameters for rule outputs
        self.rule_params = nn.Parameter(torch.randn(num_rules, input_dim + 1, num_outputs))
        # Shape: [num_rules, input_dim + 1 (bias), num_outputs]

    def forward(self, x):
        """
        Forward pass for ANFIS.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_outputs].
        """
        batch_size = x.shape[0]

        # Step 1: Rule Activation Layer
        # Compute activation of each rule (weighted sum + bias, then pass through activation)
        rule_activations = torch.tanh(torch.matmul(x, self.rule_weights.T) + self.rule_biases)
        # Shape: [batch_size, num_rules]

        # Step 2: Normalization Layer
        normalized_activations = rule_activations / torch.sum(rule_activations, dim=1, keepdim=True)
        # Shape: [batch_size, num_rules]

        # Step 3: Defuzzification Layer
        x_with_bias = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # Add bias term, shape: [batch_size, input_dim + 1]
        rule_outputs = torch.einsum('bm,rmn->brn', normalized_activations, self.rule_params)
        # Shape: [batch_size, num_rules, num_outputs]

        # Step 4: Output Layer (Aggregate rule outputs)
        output = torch.sum(rule_outputs, dim=1)  # Shape: [batch_size, num_outputs]

        return output

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 5       # Number of input features
    num_rules = 10      # Number of fuzzy rules
    num_outputs = 2     # Number of output dimensions (e.g., multi-output)

    # Create ANFIS model
    model = ANFIS(input_dim, num_rules, num_outputs)

    # Generate some random data
    x = torch.randn(5, input_dim)  # Batch of 5 samples, each with input_dim features
    output = model(x)

    print("Input:\n", x)
    print("Output:\n", output)
