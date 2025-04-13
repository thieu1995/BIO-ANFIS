#!/usr/bin/env python
# Created by "Thieu" at 04:33, 26/10/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


class MlpNet:
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_rate=0.5):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01 for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

    def relu(self, z):
        return np.maximum(0, z)

    def apply_dropout(self, layer_output):
        if self.dropout_rate > 0:
            mask = np.random.rand(*layer_output.shape) > self.dropout_rate
            return layer_output * mask
        return layer_output

    def forward(self, X):
        self.activations = []
        self.z_values = []
        for i in range(len(self.weights)):
            z = np.dot(X if i == 0 else self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.relu(z) if i < len(self.weights) - 1 else z
            activation = self.apply_dropout(activation)
            self.activations.append(activation)
        return self.activations[-1]

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, input_size, hidden_layer_sizes, output_size):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.g_best = None
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize a population of random weights and biases
        population = []
        for _ in range(self.population_size):
            mlp = MlpNet(self.input_size, self.hidden_layer_sizes, self.output_size)
            individual = self.get_flattened_weights_and_biases(mlp)
            population.append(individual)
        return np.array(population)

    def get_flattened_weights_and_biases(self, mlp):
        return np.concatenate([w.flatten() for w in mlp.weights] + [b.flatten() for b in mlp.biases])

    def fitness(self, individual, X, y):
        # Create an MLP from the individual weights and biases
        mlp = self.reconstruct_mlp(individual)
        predictions = mlp.forward(X)
        # Use mean squared error or accuracy as the fitness score
        # Here we assume y is one-hot encoded
        return -np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))  # Higher is better

    def reconstruct_mlp(self, individual):
        # Convert the flat individual back to weights and biases
        mlp = MlpNet(self.input_size, self.hidden_layer_sizes, self.output_size)
        start = 0
        for i in range(len(mlp.weights)):
            end = start + mlp.weights[i].size
            mlp.weights[i] = individual[start:end].reshape(mlp.weights[i].shape)
            start = end
        for i in range(len(mlp.biases)):
            end = start + mlp.biases[i].size
            mlp.biases[i] = individual[start:end].reshape(mlp.biases[i].shape)
            start = end
        return mlp

    # def selection(self, fitness_scores):
    #     # Select individuals based on fitness (tournament selection)
    #     idx = np.random.choice(np.arange(self.population_size), size=self.population_size, replace=True)
    #     selected = self.population[idx]
    #     selected_fitness = fitness_scores[idx]
    #     return selected[np.argsort(-selected_fitness)][:self.population_size // 2]  # Top half

    def selection(self, fitness_scores):
        # Select indices within the range of the current population size
        idx = np.random.choice(np.arange(len(self.population)), size=len(self.population), replace=True)
        selected = self.population[idx]
        selected_fitness = fitness_scores[idx]
        # Sort based on fitness and select the top half
        selected_sorted = selected[np.argsort(-selected_fitness)][:self.population_size // 2]
        return selected_sorted

    def crossover(self, parent1, parent2):
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child

    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.randn() * 0.1  # Small random mutation
        return individual

    def evolve(self, X, y, generations):
        fitness_scores = np.array([self.fitness(ind, X, y) for ind in self.population])
        self.g_best = np.max(fitness_scores)
        for generation in range(generations):
            selected_population = self.selection(fitness_scores)
            next_generation = []
            for i in range(len(selected_population)):
                parent1 = selected_population[i]
                parent2 = selected_population[np.random.randint(len(selected_population))]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            self.population = np.array(next_generation)
            fitness_scores = np.array([self.fitness(ind, X, y) for ind in self.population])
            self.g_best = np.max(fitness_scores.tolist() + [self.g_best])
            print(f"Epoch: {generation}, Best fit: {self.g_best}")


# Example usage
if __name__ == "__main__":
    # Example dataset (you should replace this with real data)
    num_samples = 100
    input_size = 784
    hidden_layer_sizes = [128, 64]
    output_size = 10
    X = np.random.randn(num_samples, input_size)  # Dummy input
    y = np.eye(output_size)[np.random.choice(output_size, num_samples)]  # Dummy one-hot encoded output

    # Genetic Algorithm settings
    population_size = 10
    mutation_rate = 0.1
    generations = 50

    # Create and run the Genetic Algorithm
    ga = GeneticAlgorithm(population_size, mutation_rate, input_size, hidden_layer_sizes, output_size)
    ga.evolve(X, y, generations)
