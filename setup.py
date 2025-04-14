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
    name="xanfis",
    version="0.1.0",
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="X-ANFIS: An Extensible and Cross-Learning ANFIS Framework for Machine Learning Tasks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=[
        "adaptive neuro-fuzzy inference system", "ANFIS", "neuro-fuzzy system",
        "fuzzy logic", "fuzzy inference", "membership functions", "Gaussian membership function",
        "bell-shaped membership function", "triangular membership function", "Takagi-Sugeno fuzzy model",
        "multi-input multi-output (MIMO)", "multi-rule fuzzy model", "rule-based system",
        "hybrid learning", "gradient descent", "least squares estimation", "backpropagation",
        "fuzzy rule learning", "rule extraction", "vectorized implementation", "pytorch fuzzy system",
        "deep fuzzy system", "differentiable fuzzy system", "interpretable learning", "explainable AI (XAI)",
        "bio-inspired optimization", "metaheuristic tuning", "neuro-fuzzy modeling", "evolutionary learning",
        "machine learning", "regression", "classification", "time series forecasting",
        "soft computing", "computational intelligence", "intelligent systems",
        "genetic algorithm (GA)", "particle swarm optimization (PSO)", "grey wolf optimizer (GWO)",
        "whale optimization algorithm (WOA)", "artificial bee colony (ABC)", "ant colony optimization (ACO)",
        "differential evolution (DE)", "simulated annealing", "bio-inspired optimization",
        "convergence analysis", "performance evaluation", "parameter optimization",
        "search space exploration", "fuzzy neural network", "robust fuzzy modeling",
        "intelligent decision system", "adaptive system", "simulation studies"
    ],
    url="https://github.com/thieu1995/X-ANFIS",
    project_urls={
        'Documentation': 'https://xanfis.readthedocs.io/',
        'Source Code': 'https://github.com/thieu1995/X-ANFIS',
        'Bug Tracker': 'https://github.com/thieu1995/X-ANFIS/issues',
        'Change Log': 'https://github.com/thieu1995/X-ANFIS/blob/main/ChangeLog.md',
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
