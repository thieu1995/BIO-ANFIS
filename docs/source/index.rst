.. MetaPerceptron documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MetaPerceptron's documentation!
====================================

.. image:: https://img.shields.io/badge/release-1.1.0-yellow.svg
   :target: https://github.com/thieu1995/metaperceptron/releases

.. image:: https://img.shields.io/pypi/wheel/gensim.svg
   :target: https://pypi.python.org/pypi/metaperceptron

.. image:: https://badge.fury.io/py/metaperceptron.svg
   :target: https://badge.fury.io/py/metaperceptron

.. image:: https://img.shields.io/pypi/pyversions/metaperceptron.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/status/metaperceptron.svg
   :target: https://img.shields.io/pypi/status/metaperceptron.svg

.. image:: https://img.shields.io/pypi/dm/metaperceptron.svg
   :target: https://img.shields.io/pypi/dm/metaperceptron.svg

.. image:: https://github.com/thieu1995/metaperceptron/actions/workflows/publish-package.yaml/badge.svg
   :target: https://github.com/thieu1995/metaperceptron/actions/workflows/publish-package.yaml

.. image:: https://pepy.tech/badge/metaperceptron
   :target: https://pepy.tech/project/metaperceptron

.. image:: https://img.shields.io/github/release-date/thieu1995/metaperceptron.svg
   :target: https://img.shields.io/github/release-date/thieu1995/metaperceptron.svg

.. image:: https://readthedocs.org/projects/metaperceptron/badge/?version=latest
   :target: https://metaperceptron.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/github/contributors/thieu1995/metaperceptron.svg
   :target: https://img.shields.io/github/contributors/thieu1995/metaperceptron.svg

.. image:: https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10067995.svg
   :target: https://doi.org/10.5281/zenodo.10067995

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


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

.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/metaperceptron.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
