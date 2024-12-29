#!/usr/bin/env python
# Created by "Thieu" at 13:24, 25/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from setuptools import setup, find_packages


def readme():
    with open('README.md', encoding='utf-8') as f:
        README = f.read()
    return README


setup(
    name="banfis",
    version="0.1.0",
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="BIO-ANFIS: A Bio-inspired Optimization Framework for Improving ANFIS Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=["multi-layer perceptron", "machine learning", "artificial intelligence",
              "deep learning", "neural networks", "single hidden layer network",
              "random projection", "FLANN", "feed-forward neural network", "artificial neural network",
              "classification", "regression", "supervised learning", "online learning", "generalization",
              "optimization algorithms", "Kernel MLP", "Cross-validation"
              "Genetic algorithm (GA)", "Particle swarm optimization (PSO)", "Ant colony optimization (ACO)",
              "Differential evolution (DE)", "Simulated annealing", "Grey wolf optimizer (GWO)",
              "Whale Optimization Algorithm (WOA)", "confusion matrix", "recall", "precision", "accuracy",
              "pearson correlation coefficient (PCC)", "spearman correlation coefficient (SCC)",
              "Global optimization", "Convergence analysis", "Search space exploration", "Local search",
              "Computational intelligence", "Robust optimization", "metaheuristic", "metaheuristic algorithms",
              "nature-inspired computing", "nature-inspired algorithms", "swarm-based computation",
              "metaheuristic-based multi-layer perceptron", "metaheuristic-optimized MLP",
              "Performance analysis", "Intelligent optimization", "Simulations"],
    url="https://github.com/thieu1995/MetaPerceptron",
    project_urls={
        'Documentation': 'https://metaperceptron.readthedocs.io/',
        'Source Code': 'https://github.com/thieu1995/MetaPerceptron',
        'Bug Tracker': 'https://github.com/thieu1995/MetaPerceptron/issues',
        'Change Log': 'https://github.com/thieu1995/MetaPerceptron/blob/main/ChangeLog.md',
        'Forum': 'https://t.me/+fRVCJGuGJg1mNDg1',
    },
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=["numpy>=1.17.1", "scipy>=1.7.1", "scikit-learn>=1.2.1", "pandas>=1.3.5",
                      "mealpy>=3.0.1", "permetrics>=2.0.0", "torch>=2.0.0"],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov==4.0.0", "flake8>=4.0.1"],
    },
    python_requires='>=3.8',
)
