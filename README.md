
[![GitHub release](https://img.shields.io/badge/release-1.1.0-yellow.svg)](https://github.com/thieu1995/MetaPerceptron/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/metaperceptron) 
[![PyPI version](https://badge.fury.io/py/metaperceptron.svg)](https://badge.fury.io/py/metaperceptron)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metaperceptron.svg)
![PyPI - Status](https://img.shields.io/pypi/status/metaperceptron.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/metaperceptron.svg)
[![Downloads](https://pepy.tech/badge/metaperceptron)](https://pepy.tech/project/metaperceptron)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/metaperceptron/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/metaperceptron/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/metaperceptron.svg)
[![Documentation Status](https://readthedocs.org/projects/metaperceptron/badge/?version=latest)](https://metaperceptron.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/metaperceptron.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/676088001.svg)](https://zenodo.org/doi/10.5281/zenodo.10251021)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


MetaPerceptron (Metaheuristic-optimized Multi-Layer Perceptron) is a Python library that implements variants and the 
traditional version of Multi-Layer Perceptron models. These include Metaheuristic-optimized MLP models (GA, PSO, WOA, TLO, DE, ...) 
and Gradient Descent-optimized MLP models (SGD, Adam, Adelta, Adagrad, ...). It provides a comprehensive list of 
optimizers for training MLP models and is also compatible with the Scikit-Learn library. With MetaPerceptron, 
you can perform searches and hyperparameter tuning using the features provided by the Scikit-Learn library.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: MlpRegressor, MlpClassifier, MhaMlpRegressor, MhaMlpClassifier
* **Total Metaheuristic-based MLP Regressor**: > 200 Models 
* **Total Metaheuristic-based MLP Classifier**: > 200 Models
* **Total Gradient Descent-based MLP Regressor**: 12 Models
* **Total Gradient Descent-based MLP Classifier**: 12 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions (as fitness functions or loss functions)**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://metaperceptron.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, torch, skorch


# Citation Request 

If you want to understand how Metaheuristic is applied to Multi-Layer Perceptron, you need to read the paper 
titled **"Let a biogeography-based optimizer train your Multi-Layer Perceptron"**. 
The paper can be accessed at the following [link](https://doi.org/10.1016/j.ins.2014.01.038)


Please include these citations if you plan to use this library:

```code

@software{nguyen_van_thieu_2023_10251022,
  author       = {Nguyen Van Thieu},
  title        = {BIO-ANFIS: A Bio-inspired Optimization Framework for Improving ANFIS Networks},
  month        = dec,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10251021},
  url          = {https://github.com/thieu1995/MetaPerceptron}
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}

@article{van2023groundwater,
  title={Groundwater level modeling using Augmented Artificial Ecosystem Optimization},
  author={Van Thieu, Nguyen and Barma, Surajit Deb and Van Lam, To and Kisi, Ozgur and Mahesha, Amai},
  journal={Journal of Hydrology},
  volume={617},
  pages={129034},
  year={2023},
  publisher={Elsevier}
}

@article{thieu2019efficient,
  title={Efficient time-series forecasting using neural network and opposition-based coral reefs optimization},
  author={Thieu Nguyen, Tu Nguyen and Nguyen, Binh Minh and Nguyen, Giang},
  journal={International Journal of Computational Intelligence Systems},
  volume={12},
  number={2},
  pages={1144--1161},
  year={2019}
}

```

# General tutorial

Below tutorial is how you can install and use this library. For more complex examples and documentation please check the examples folder and documentation website.

### The OOP structure idea
The structure and idea of this library is followed:

```code
.CustomMLP class: Define general Pytorch model

.BaseMlp class: Inherit BaseEstimator from Scikit-Learn
    + Contains CustomMLP object
    
    .BaseStandardMlp class: Inherit BaseMlp class       
        + Gradient-based training MLP wrapper for both regression and classification
        
        .MlpRegressor class: Inherit BaseStandardMlp and RegressorMixin classes
            + Gradient-based training MLP for regression tasks
        
        .MlpClassifier class: Inherit BaseStandardMlp and ClassifierMixin classes
            + Gradient-based training MLP for classification tasks

    .BaseMhaMlp class: Inherit BaseMlp class
        + Metaheuristic-based training MLP wrapper for both regregression and classification
        
        .MhaMlpRegressor class: Inherit BaseMhaMlp and RegressorMixin classes
            + Metaheuristic-based training MLP for regression tasks
            
        .MhaMlpClassifier class: Inherit BaseMhaMlp and ClassifierMixin classes
            + Metaheuristic-based training MLP for classification tasks
```

### Installation

* Install the [current PyPI release](https://pypi.python.org/pypi/metaperceptron):
```sh 
$ pip install banfis==1.1.0
```

### Check version

After installation, you can import MetaPerceptron as any other Python module:

```sh
$ python
>>> import banfis
>>> banfis.__version__
```

### General example

Let's say I want to use Genetic Algorithm-based MLP for Iris classification dataset. Here how to do it.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from banfis import MhaMlpClassifier

# Load and prepare the dataset
data = load_iris()
X = data.data
y = data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the object with hyper-parameters
net = MhaMlpClassifier(hidden_layers=(100,), act_names="ELU", dropout_rates=None, act_output=None,
                       optim="BaseGA", optim_paras=None, obj_name="F1S", seed=42, verbose=True)

# Fit (train) the model
net.fit(X_train, y_train, mode="swarm", n_workers=6)

# Score the model on test set
res = net.score(X_test, y_test)

print(res)
print(net.model)
print(net.model.get_weights())
```

As can be seen, you do it like any model from Scikit-Learn library such as SVC, RF, DT,...




* Install directly from source code
```sh 
$ git clone https://github.com/thieu1995/MetaPerceptron.git
$ cd MetaPerceptron
$ python setup.py install
```

* In case, you want to install the development version from Github:
```sh 
$ pip install git+https://github.com/thieu1995/MetaPerceptron 
```

After installation, you can import MetaPerceptron as any other Python module:

```sh
$ python
>>> import banfis
>>> banfis.__version__
```

### Examples

Please check all use cases and examples in folder [examples](examples).

1) MetaPerceptron provides this useful classes

```python
from banfis import DataTransformer, Data
from banfis import MlpRegressor, MlpClassifier
from banfis import MhaMlpRegressor, MhaMlpClassifier
```

2) What you can do with `DataTransformer` class

We provide many scaler classes that you can select and make a combination of transforming your data via 
DataTransformer class. For example: 

2.1) I want to scale data by `Loge` and then `Sqrt` and then `MinMax`:

```python
from banfis import DataTransformer
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

dt = DataTransformer(scaling_methods=("loge", "sqrt", "minmax"))
X_train_scaled = dt.fit_transform(X_train)
X_test_scaled = dt.transform(X_test)
```

2.2) I want to scale data by `YeoJohnson` and then `Standard`:

```python
from banfis import DataTransformer
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

dt = DataTransformer(scaling_methods=("yeo-johnson", "standard"))
X_train_scaled = dt.fit_transform(X_train)
X_test_scaled = dt.transform(X_test)
```

3) What can you do with `Data` class
+ You can load your dataset into Data class
+ You can split dataset to train and test set
+ You can scale dataset without using DataTransformer class
+ You can scale labels using LabelEncoder

```python
from banfis import Data
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

data = Data(X, y, name="position_salaries")

#### Split dataset into train and test set
data.split_train_test(test_size=0.2, shuffle=True, random_state=100, inplace=True)

#### Feature Scaling
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "sqrt", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
data.y_test = scaler_y.transform(data.y_test)
```

4) What can you do with all model classes
+ Define the model 
+ Use provides functions to train, predict, and evaluate model

```python
from banfis import MlpRegressor, MlpClassifier, MhaMlpRegressor, MhaMlpClassifier

## Use standard MLP model for regression problem
regressor = MlpRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
                         max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)

## Use standard MLP model for classification problem 
classifier = MlpClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
                           max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)

## Use Metaheuristic-optimized MLP model for regression problem
print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
print(MhaMlpClassifier.SUPPORTED_REG_OBJECTIVES)

opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
regressor = MhaMlpRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
                            obj_name="MSE", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)

## Use Metaheuristic-optimized MLP model for classification problem
print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
print(MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES)

opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
classifier = MhaMlpClassifier(hidden_size=50, act1_name="tanh", act2_name="softmax",
                              obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
```

5) What can you do with model object

```python
from banfis import MlpRegressor, Data

data = Data()  # Assumption that you have provide this object like above

model = MlpRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
                     max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)

## Train the model
model.fit(data.X_train, data.y_train)

## Predicting a new result
y_pred = model.predict(data.X_test)

## Calculate metrics using score or scores functions.
print(model.score(data.X_test, data.y_test, method="MAE"))
print(model.scores(data.X_test, data.y_test, list_methods=["MAPE", "NNSE", "KGE", "MASE", "R2", "R", "R2S"]))

## Calculate metrics using evaluate function
print(model.evaluate(data.y_test, y_pred, list_metrics=("MSE", "RMSE", "MAPE", "NSE")))

## Save performance metrics to csv file
model.save_evaluation_metrics(data.y_test, y_pred, list_metrics=("RMSE", "MAE"), save_path="history",
                              filename="metrics.csv")

## Save training loss to csv file
model.save_training_loss(save_path="history", filename="loss.csv")

## Save predicted label
model.save_y_predicted(X=data.X_test, y_true=data.y_test, save_path="history", filename="y_predicted.csv")

## Save model
model.save_model(save_path="history", filename="traditional_mlp.pkl")

## Load model 
trained_model = MlpRegressor.load_model(load_path="history", filename="traditional_mlp.pkl")
```

# Support (questions, problems)

### Official Links 

* Official source code repo: https://github.com/thieu1995/MetaPerceptron
* Official document: https://metapeceptron.readthedocs.io/
* Download releases: https://pypi.org/project/metaperceptron/
* Issue tracker: https://github.com/thieu1995/MetaPerceptron/issues
* Notable changes log: https://github.com/thieu1995/MetaPerceptron/blob/master/ChangeLog.md
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
