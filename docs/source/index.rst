.. XANFIS documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XANFIS's documentation!
==================================

.. image:: https://img.shields.io/badge/release-1.0.0-yellow.svg
   :target: https://github.com/thieu1995/X-ANFIS/releases

.. image:: https://img.shields.io/pypi/wheel/gensim.svg
   :target: https://pypi.python.org/pypi/xanfis

.. image:: https://badge.fury.io/py/xanfis.svg
   :target: https://badge.fury.io/py/xanfis

.. image:: https://img.shields.io/pypi/pyversions/xanfis.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/status/xanfis.svg
   :target: https://img.shields.io/pypi/status/xanfis.svg

.. image:: https://img.shields.io/pypi/dm/xanfis.svg
   :target: https://img.shields.io/pypi/dm/xanfis.svg

.. image:: https://github.com/thieu1995/X-ANFIS/actions/workflows/publish-package.yaml/badge.svg
   :target: https://github.com/thieu1995/X-ANFIS/actions/workflows/publish-package.yaml

.. image:: https://pepy.tech/badge/xanfis
   :target: https://pepy.tech/project/xanfis

.. image:: https://img.shields.io/github/release-date/thieu1995/X-ANFIS.svg
   :target: https://img.shields.io/github/release-date/thieu1995/X-ANFIS.svg

.. image:: https://readthedocs.org/projects/xanfis/badge/?version=latest
   :target: https://xanfis.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/github/contributors/thieu1995/X-ANFIS.svg
   :target: https://img.shields.io/github/contributors/thieu1995/X-ANFIS.svg

.. image:: https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10067995.svg
   :target: https://doi.org/10.5281/zenodo.10067995

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


**X-ANFIS** (An Extensible and Cross-Learning ANFIS Framework for Machine Learning Tasks) is a Python
framework designed for Adaptive Neuro-Fuzzy Inference Systems with high customizability and modern ML integration.
X-ANFIS is fully modular, written in PyTorch, and compatible with Scikit-Learn pipelines.
It supports a wide range of learning strategies, including gradient descent, least squares estimation,
and even bio-inspired methods.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimators**: AnfisRegressor, AnfisClassifier, GdAnfisRegressor, GdAnfisClassifier, BioAnfisRegressor, BioAnfisClassifier
* **Supported Membership Functions**: Gaussian, Bell, Triangular, Custom
* **Supported Learning Strategies**: Hybrid (GD + LSE), Gradient-only, Bio-inspired (Bio + LSE)
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions (as fitness functions or loss functions)**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://xanfis.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, torch


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/xanfis.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
