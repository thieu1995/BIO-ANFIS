#!/usr/bin/env python
# Created by "Thieu" at 21:53, 29/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn

class ANFIS(nn.Module):
    def __init__(self, input_dim, num_mfs, num_rules, num_outputs):
        """
        ANFIS model with configurable number of rules.

        Args:
            input_dim (int): Number of input features.
            num_mfs (int): Number of membership functions per input.
            num_rules (int): Number of fuzzy rules.
            num_outputs (int): Number of output dimensions.
        """
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        self.num_rules = num_rules
        self.num_outputs = num_outputs

        # Parameters for membership functions (Gaussian)
        self.centers = nn.Parameter(torch.randn(input_dim, num_mfs))  # Center of each MF
        self.widths = nn.Parameter(torch.ones(input_dim, num_mfs))    # Width of each MF

        # Rule parameters
        self.rule_indices = nn.Parameter(torch.randint(0, num_mfs, (num_rules, input_dim)), requires_grad=False)
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

        # Step 1: Fuzzification Layer (Gaussian Membership Functions)
        mfs = []
        for i in range(self.input_dim):
            mf = torch.exp(-((x[:, i].unsqueeze(1) - self.centers[i]) ** 2) / (2 * self.widths[i] ** 2))
            mfs.append(mf)  # Shape: [batch_size, num_mfs]
        mfs = torch.stack(mfs, dim=1)  # Shape: [batch_size, input_dim, num_mfs]

        # Step 2: Rule Layer (AND operation using rule_indices)
        # Select relevant membership functions for each rule
        selected_mfs = mfs.gather(2, self.rule_indices.unsqueeze(0).expand(batch_size, -1, -1))
        # Shape: [batch_size, input_dim, num_rules]

        rule_activations = torch.prod(selected_mfs, dim=1)  # Shape: [batch_size, num_rules]

        # Step 3: Normalization Layer
        normalized_activations = rule_activations / torch.sum(rule_activations, dim=1, keepdim=True)  # Shape: [batch_size, num_rules]

        # Step 4: Defuzzification Layer (Linear combination for each rule)
        x_with_bias = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # Shape: [batch_size, input_dim + 1]
        rule_outputs = torch.einsum('bm,rmn->brn', normalized_activations, self.rule_params)  # Shape: [batch_size, num_rules, num_outputs]

        # Step 5: Output Layer (Aggregate rule outputs)
        output = torch.sum(rule_outputs, dim=1)  # Shape: [batch_size, num_outputs]

        return output

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 5       # Number of input features
    num_mfs = 3         # Number of membership functions per input
    num_rules = 10      # Number of fuzzy rules
    num_outputs = 2     # Number of output dimensions (e.g., multi-output)

    # Create ANFIS model
    model = ANFIS(input_dim, num_mfs, num_rules, num_outputs)

    # Generate some random data
    x = torch.randn(5, input_dim)  # Batch of 5 samples, each with input_dim features
    output = model(x)

    print("Input:\n", x)
    print("Output:\n", output)
