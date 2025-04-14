#!/usr/bin/env python
# Created by "Thieu" at 14:24, 26/10/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from xanfis.core.base_mlp import CustomMLP


def check_CustomMLP_class():
    # Example usage
    input_size = 10
    output_size = 2
    hidden_layers = [64, 32, 16]  # Three hidden layers with specified nodes
    activations = ["ReLU", "Tanh", "ReLU"]  # Activation functions for each layer
    dropouts = [0.2, 0.3, 0.0]  # Dropout rates for each hidden layer

    model = CustomMLP(input_size, output_size, hidden_layers, activations, dropouts)
    print(model)


check_CustomMLP_class()
