# X-ANFIS: Modular ANFIS Library with Classic, Gradient, and Bio-inspired Training Strategies

[![GitHub release](https://img.shields.io/badge/release-1.1.0-yellow.svg)](https://github.com/thieu1995/X-ANFIS/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/xanfis) 
[![PyPI version](https://badge.fury.io/py/xanfis.svg)](https://badge.fury.io/py/xanfis)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xanfis.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/xanfis.svg)
[![Downloads](https://pepy.tech/badge/xanfis)](https://pepy.tech/project/xanfis)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/X-ANFIS/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/X-ANFIS/actions/workflows/publish-package.yaml)
[![Documentation Status](https://readthedocs.org/projects/xanfis/badge/?version=latest)](https://x-anfis.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.28802531-blue)](https://doi.org/10.6084/m9.figshare.28802531)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

**X-ANFIS** is a Python library offering a powerful and extensible implementation of Adaptive Neuro-Fuzzy Inference System (ANFIS) 
using **PyTorch** and **Scikit-Learn**. The library is written with object-oriented principles and modular architecture, 
enabling easy customization, integration, and experimentation.It supports:

- 🔧 **Classic ANFIS** (Hybrid: Gradient Descent + Least Squares)
- 🌊 **Gradient-based ANFIS** (Fully differentiable training)
- 🧠 **Bio-inspired ANFIS** (Metaheuristics + Least Squares)

## ✨ Key Features

- Modular, object-oriented design for ease of extension.
- **Scikit-Learn API:** `.fit()`, `.predict()`, `.score()` and compatible with `GridSearchCV`, `Pipeline`, etc.
- Wide range of Gradient-based training optimizers: `SGD`, `Adam`, `RMSprop`, `Adagrad`, `AdamW`, ...
- Wide range of Bio-inspired optimizers: `GA`, `PSO`, `ACO`, `WOA`, `BA`, `FFA`, `CSA`, `GWO`, `ABC`, `DE`, ...
- Rich membership functions: `Triangular`, `Gaussian`, `Bell`, ...
- Over 67 built-in metrics and losses.

## 🧠 Model Zoo

| Model Class                               | Training Method     | Type                        |
|-------------------------------------------|---------------------|-----------------------------|
| `AnfisRegressor`, `AnfisClassifier`       | Classic Hybrid      | Regression / Classification |
| `GdAnfisRegressor`, `GdAnfisClassifier`   | Gradient-based      | Regression / Classification |
| `BioAnfisRegressor`, `BioAnfisClassifier` | Metaheuristic-based | Regression / Classification | 


## 📌 Citation

Please include these citations if you plan to use this library:

```bibtex
@software{thieu20250414,
  author  = {Nguyen Van Thieu},
  title   = {X-ANFIS: An Extensible and Cross-Learning ANFIS Framework for Machine Learning Tasks},
  month   = June,
  year    = 2025,
  doi     = {10.6084/m9.figshare.28802531},
  url     = {https://github.com/thieu1995/X-ANFIS}
}

@article{van2023mealpy,
  title   = {MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author  = {Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal = {Journal of Systems Architecture},
  year    = {2023},
  publisher = {Elsevier},
  doi     = {10.1016/j.sysarc.2023.102871}
}
```

# ⚙️ General tutorial

Below tutorial is how you can install and use this library. For more complex examples and documentation 
please check the [examples](examples) folder and documentation website.


## 📦 Installation

Install the latest version using pip:

```bash
pip install xanfis
```

After that, check the version to ensure successful installation:

```bash
$ python
>>> import xanfis
>>> xanfis.__version__
```


## 🧪 Quick Example

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


## 💬 Contributing

We welcome contributions to X-ANFIS! If you have suggestions, improvements, or bug fixes, feel free to fork 
the repository, create a pull request, or open an issue.

X-ANFIS allows for customization at various levels, including:

* Membership Functions: You can define custom membership functions to fit your problem requirements.

* Learning Strategies: Easily switch between gradient-based or bio-inspired algorithms for training.

* Model Components: Customize the architecture of the ANFIS model, including the number of input and output nodes, 
output activation function, number of rules, and rule strengths, L2 regularization, training methods.


## 📞 Community & Support

- 📖 [Official Source Code](https://github.com/thieu1995/X-ANFIS)
- 📖 [Official Releases](https://pypi.org/project/xanfis/)
- 📖 [Official Docs](https://x-anfis.readthedocs.io/)
- 💬 [Telegram Chat](https://t.me/+fRVCJGuGJg1mNDg1)
- 🐛 [Report Issues](https://github.com/thieu1995/X-ANFIS/issues)
- 🔄 [Changelog](https://github.com/thieu1995/X-ANFIS/blob/master/ChangeLog.md)


## 🧩 Related Projects

Explore other projects by the author:

- 🔧 [MEALPY](https://github.com/thieu1995/mealpy)
- 🔍 [Metaheuristics](https://github.com/thieu1995/metaheuristics)
- 🧪 [Permetrics](https://github.com/thieu1995/permetrics)
- 📦 [Opfunu](https://github.com/thieu1995/opfunu)
- 🔬 [PFEvaluator](https://github.com/thieu1995/pfevaluator)
- 🧠 [IntelELM](https://github.com/thieu1995/IntelELM)
- 🔥 [Reflame](https://github.com/thieu1995/reflame)
- 🧭 [MetaCluster](https://github.com/thieu1995/MetaCluster)
- 🧠 [Enoppy](https://github.com/thieu1995/enoppy)
- 🤖 [AIIR Team](https://github.com/aiir-team)

---

Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=XANFIS_QUESTIONS) @ 2025
