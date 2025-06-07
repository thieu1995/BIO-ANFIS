#!/usr/bin/env python
# Created by "Thieu" at 07:14, 14/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from xanfis import Data, AnfisRegressor
from sklearn.datasets import load_diabetes


## Load data object
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
data.y_test = scaler_y.transform(data.y_test)

print(type(data.X_train), type(data.y_train))

## Create model
model = AnfisRegressor(num_rules=10, mf_class="Gaussian", vanishing_strategy="prod",
                       act_output=None, reg_lambda=0.001,
                       epochs=100, batch_size=16, optim="Adam", optim_params=None,
                       early_stopping=True, n_patience=30, epsilon=0.1, valid_rate=0.1,
                       seed=42, verbose=True, device="cpu")
## Train the model
model.fit(data.X_train, data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
# print(y_pred)

## Calculate some metrics
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["R", "NSE", "MAPE", "KGE", "R2S", "R2"]))
