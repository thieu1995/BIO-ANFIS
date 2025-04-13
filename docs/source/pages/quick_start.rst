============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/metaperceptron />`_::

   $ pip install metaperceptron==1.1.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/metaperceptron.git
   $ cd metaperceptron
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/metaperceptron


After installation, you can import MetaPerceptron as any other Python module::

   $ python
   >>> import metaperceptron
   >>> metaperceptron.__version__

========
Examples
========

In this section, we will explore the usage of the MetaPerceptron model with the assistance of a dataset. While all the
preprocessing steps mentioned below can be replicated using Scikit-Learn, we have implemented some utility functions
to provide users with convenience and faster usage.


**Combine MetaPerceptron library like a normal library with scikit-learn**::

	### Step 1: Importing the libraries
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import MinMaxScaler, LabelEncoder
	from metaperceptron import MlpRegressor, MlpClassifier, MhaMlpRegressor, MhaMlpClassifier

	#### Step 2: Reading the dataset
	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:2].values
	y = dataset.iloc[:, 2].values

	#### Step 3: Next, split dataset into train and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=100)

	#### Step 4: Feature Scaling
	scaler_X = MinMaxScaler()
	scaler_X.fit(X_train)
	X_train = scaler_X.transform(X_train)
	X_test = scaler_X.transform(X_test)

	le_y = LabelEncoder()  # This is for classification problem only
	le_y.fit(y)
	y_train = le_y.transform(y_train)
	y_test = le_y.transform(y_test)

	#### Step 5: Fitting MLP-based model to the dataset

	##### 5.1: Use standard MLP model for regression problem
	regressor = MlpRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
                 max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)
	regressor.fit(X_train, y_train)

	##### 5.2: Use standard MLP model for classification problem
	classifer = MlpClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
                 max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)
	classifer.fit(X_train, y_train)

	##### 5.3: Use Metaheuristic-based MLP model for regression problem
	print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
	print(MhaMlpClassifier.SUPPORTED_REG_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	model = MhaMlpRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
                 obj_name="MSE", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
	regressor.fit(X_train, y_train)

	##### 5.4: Use Metaheuristic-based MLP model for classification problem
	print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
	print(MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	classifier = MhaMlpClassifier(hidden_size=50, act1_name="tanh", act2_name="softmax",
                 obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
	classifier.fit(X_train, y_train)

	#### Step 6: Predicting a new result
	y_pred = regressor.predict(X_test)

	y_pred_cls = classifier.predict(X_test)
	y_pred_label = le_y.inverse_transform(y_pred_cls)

	#### Step 7: Calculate metrics using score or scores functions.
	print("Try my AS metric with score function")
	print(regressor.score(X_test, y_test, method="AS"))

	print("Try my multiple metrics with scores function")
	print(classifier.scores(X_test, y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))


**Utilities everything that metaperceptron provided**::

	### Step 1: Importing the libraries
	from metaperceptron import Data, MlpRegressor, MlpClassifier, MhaMlpRegressor, MhaMlpClassifier
	from sklearn.datasets import load_digits

	#### Step 2: Reading the dataset
	X, y = load_digits(return_X_y=True)
	data = Data(X, y)

	#### Step 3: Next, split dataset into train and test set
	data.split_train_test(test_size=0.2, shuffle=True, random_state=100)

	#### Step 4: Feature Scaling
	data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("minmax"))
	data.X_test = scaler_X.transform(data.X_test)

	data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
	data.y_test = scaler_y.transform(data.y_test)

	#### Step 5: Fitting MLP-based model to the dataset
	##### 5.1: Use standard MLP model for regression problem
	regressor = MlpRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="MSE",
                 max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)
	regressor.fit(data.X_train, data.y_train)

	##### 5.2: Use standard MLP model for classification problem
	classifer = MlpClassifier(hidden_size=50, act1_name="tanh", act2_name="sigmoid", obj_name="NLLL",
                 max_epochs=1000, batch_size=32, optimizer="SGD", optimizer_paras=None, verbose=False)
	classifer.fit(data.X_train, data.y_train)

	##### 5.3: Use Metaheuristic-based MLP model for regression problem
	print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
	print(MhaMlpClassifier.SUPPORTED_REG_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	model = MhaMlpRegressor(hidden_size=50, act1_name="tanh", act2_name="sigmoid",
                 obj_name="MSE", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
	regressor.fit(data.X_train, data.y_train)

	##### 5.4: Use Metaheuristic-based MLP model for classification problem
	print(MhaMlpClassifier.SUPPORTED_OPTIMIZERS)
	print(MhaMlpClassifier.SUPPORTED_CLS_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	classifier = MhaMlpClassifier(hidden_size=50, act1_name="tanh", act2_name="softmax",
                 obj_name="CEL", optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True)
	classifier.fit(data.X_train, data.y_train)

	#### Step 6: Predicting a new result
	y_pred = regressor.predict(data.X_test)

	y_pred_cls = classifier.predict(data.X_test)
	y_pred_label = scaler_y.inverse_transform(y_pred_cls)

	#### Step 7: Calculate metrics using score or scores functions.
	print("Try my AS metric with score function")
	print(regressor.score(data.X_test, data.y_test, method="AS"))

	print("Try my multiple metrics with scores function")
	print(classifier.scores(data.X_test, data.y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))


A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize
the data within a particular range.

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
