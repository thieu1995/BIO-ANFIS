#!/usr/bin/env python
# Created by "Thieu" at 09:56, 14/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from xanfis import Data, BioAnfisClassifier
from sklearn.datasets import load_breast_cancer


## Load data object
X, y = load_breast_cancer(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True, shuffle=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)
data.y_test = scaler_y.transform(data.y_test)

## Create model
model = BioAnfisClassifier(num_rules=10, mf_class="Gaussian", act_output=None,
                           vanishing_strategy="blend", reg_lambda=None,
                           optim="BaseGA", optim_params={"name": "WOA", "epoch": 100, "pop_size": 30},
                           obj_name="F1S", seed=42, verbose=True)
## Train the model
model.fit(X=data.X_train, y=data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)

## Calculate some metrics
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["AS", "F1S", "PS", "FBS"]))
