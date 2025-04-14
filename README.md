
[![GitHub release](https://img.shields.io/badge/release-1.0.0-yellow.svg)](https://github.com/thieu1995/X-ANFIS/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/xanfis) 
[![PyPI version](https://badge.fury.io/py/xanfis.svg)](https://badge.fury.io/py/xanfis)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xanfis.svg)
![PyPI - Status](https://img.shields.io/pypi/status/xanfis.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/xanfis.svg)
[![Downloads](https://pepy.tech/badge/xanfis)](https://pepy.tech/project/xanfis)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/X-ANFIS/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/X-ANFIS/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/X-ANFIS.svg)
[![Documentation Status](https://readthedocs.org/projects/xanfis/badge/?version=latest)](https://xanfis.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/X-ANFIS.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/676088001.svg)](https://zenodo.org/doi/10.5281/zenodo.10251021)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


# X-ANFIS (Modular ANFIS Library with Classic, Gradient, and Bio-inspired Training Strategies)

X-ANFIS is a Python library that provides a flexible and extensible implementation of Adaptive Neuro-Fuzzy 
Inference System (ANFIS) models using PyTorch and Scikit-Learn APIs. The library supports multiple training strategies including:

* Classic ANFIS with hybrid learning (Gradient Descent + Least Squares Estimation)

* Gradient-based ANFIS (fully end-to-end gradient training)

* Bio-inspired ANFIS (Bio-inspired algorithms + Least Squares Estimation)

The library is written with object-oriented principles and modular architecture, enabling easy customization, integration, and experimentation.


* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: 
  * Traditional models: AnfisRegressor, AnfisClassifier
  * Full GD-based models: GdAnfisRegressor, GdAnfisClassifier
  * Bio-based models: BioAnfisRegressor, BioAnfisClassifier
* **Supported Membership Functions**: Triangular, Gaussian, Bell, and more
* **Scikit-Learn Compatible**: Supports .fit(), .predict(), .score() and works with GridSearchCV, Pipeline, etc.
* **Supported optimizers**: SGD, Adam, RMSprop, Adagrad, Adadelta, AdamW, and more
* **Supported activation functions**: ELU, SELU, GELU, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, and more
* **Supported optimizers for bio-based models**: 
  * Genetic Algorithm (GA)
  * Particle Swarm Optimization (PSO)
  * Ant Colony Optimization (ACO)
  * Whale Optimization Algorithm (WOA)
  * Bat Algorithm (BA)
  * Firefly Algorithm (FFA)
  * Cuckoo Search Algorithm (CSA)
  * Grey Wolf Optimizer (GWO)
  * Artificial Bee Colony (ABC)
  * Differential Evolution (DE) and more
* **Supported loss functions**: MSE, MAE, RMSE, MAPE, R2, R2S, NSE, KGE, and more
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://xanfis.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, torch


# Citation Request

Please include these citations if you plan to use this library:

```code

@software{nguyen_van_thieu_2023_10251022,
  author       = {Nguyen Van Thieu},
  title        = {X-ANFIS: An Extensible and Cross-Learning ANFIS Framework for Machine Learning Tasks},
  month        = april,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10251021},
  url          = {https://github.com/thieu1995/X-ANFIS}
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}
```

# General tutorial

Below tutorial is how you can install and use this library. For more complex examples and documentation 
please check the [examples](examples) folder and documentation website.


### The OOP structure idea
The structure and idea of this library is followed:

```code
CustomANFIS (torch.nn.Module)
 └── BaseAnfis (Scikit-learn BaseEstimator)
     ├── BaseClassicAnfis
     │   ├── AnfisRegressor
     │   └── AnfisClassifier
     ├── BaseGdAnfis
     │   ├── GdAnfisRegressor
     │   └── GdAnfisClassifier
     └── BaseBioAnfis
         ├── BioAnfisRegressor
         └── BioAnfisClassifier

.CustomANFIS class: Define general Pytorch model

.BaseAnfis class: Inherit BaseEstimator from Scikit-Learn

.BaseClassicAnfis: Inherit BaseAnfis class, this is classical (traditional) ANFIS model
  + Purpose: Gradient-based training for membership parameters, and Pseudo-inverse and Ridge regresison for consequent (output weights)
  + AnfisRegressor: Inherit BaseClassicAnfis and RegressorMixin classes, ANFIS wrapper for regression tasks
  + AnfisClassifier: Inherit BaseClassicAnfis and ClassifierMixin classes, ANFIS wrapper for classification tasks

.BaseGdAnfis: Inherit BaseAnfis class, this is gradient-based ANFIS model
  + Purpose: Gradient-based training for both membership parameters and consequent (output weights)
  + GdAnfisRegressor: Inherit BaseGdAnfis and RegressorMixin classes, ANFIS wrapper for regression tasks
  + GdAnfisClassifier: Inherit BaseGdAnfis and ClassifierMixin classes, ANFIS wrapper for classification tasks

.BaseBioAnfis: Inherit BaseAnfis class, this is bio-inspired ANFIS model
  + Purpose: Bio-inspired training for both membership parameters and Psedo-inverse and Ridge regresison for consequent (output weights)
  + BioAnfisRegressor: Inherit BaseBioAnfis and RegressorMixin classes, ANFIS wrapper for regression tasks
  + BioAnfisClassifier: Inherit BaseBioAnfis and ClassifierMixin classes, ANFIS wrapper for classification tasks
```

### Installation

* Install the [current PyPI release](https://pypi.python.org/pypi/xanfis):
```sh 
$ pip install xanfis
```

### Check version

After installation, you can import X-ANFIS as any other Python module:

```sh
$ python
>>> import xanfis
>>> xanfis.__version__
```


### General example

Let's say I want to use Adam optimization-based ANFIS for Iris classification dataset. Here how to do it.

```python
from xanfis import Data, GdAnfisClassifier
from sklearn.datasets import load_iris


## Load data object
X, y = load_iris(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True, shuffle=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)
data.y_test = scaler_y.transform(data.y_test)

print(type(data.X_train), type(data.y_train))

## Create model
model = GdAnfisClassifier(num_rules=20, mf_class="Gaussian",
                          act_output=None, vanishing_strategy="blend", reg_lambda=None,
                          epochs=50, batch_size=16, optim="Adam", optim_params={"lr": 0.01},
                          early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                          seed=42, verbose=True)
## Train the model
model.fit(X=data.X_train, y=data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)
print(model.predict_proba(data.X_test))

## Calculate some metrics
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS", "PS", "RS", "NPV", "F1S"]))
```

As can be seen, you do it like any model from Scikit-Learn library such as SVC, RF, DT,...


# Support (questions, problems)

### Customization

X-ANFIS allows for customization at various levels, including:

* Membership Functions: You can define custom membership functions to fit your problem requirements.

* Learning Strategies: Easily switch between gradient-based or bio-inspired algorithms for training.

* Model Components: Customize the architecture of the ANFIS model, including the number of input and output nodes, 
output activation function, number of rules, and rule strengths, L2 regularization, training methods.


### Contributing
We welcome contributions to X-ANFIS! If you have suggestions, improvements, or bug fixes, feel free to fork 
the repository, create a pull request, or open an issue.


### Official Links 

* Official source code repo: https://github.com/thieu1995/X-ANFIS
* Official document: https://xanfis.readthedocs.io/
* Download releases: https://pypi.org/project/xanfis/
* Issue tracker: https://github.com/thieu1995/X-ANFIS/issues
* Notable changes log: https://github.com/thieu1995/X-ANFIS/blob/master/ChangeLog.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

* This project also related to our another projects which are "optimization" and "machine learning", check it here:
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/enoppy
    * https://github.com/thieu1995/permetrics
    * https://github.com/thieu1995/MetaCluster
    * https://github.com/thieu1995/pfevaluator
    * https://github.com/thieu1995/IntelELM
    * https://github.com/thieu1995/reflame
    * https://github.com/aiir-team
