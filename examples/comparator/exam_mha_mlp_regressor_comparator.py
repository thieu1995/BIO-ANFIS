#!/usr/bin/env python
# Created by "Thieu" at 19:59, 19/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_diabetes
from banfis import MhaMlpComparator


## Load data object
X, y = load_diabetes(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape)

# Min-max the output
y_scaler = MinMaxScaler(feature_range=(0, 1))
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test = y_scaler.transform(y_test.reshape(-1, 1))


# Here is the list of optimizers you want to compare
optim_dict = {
    'BaseGA':       {"epoch": 10, "pop_size": 20},
    "OriginalPSO":  {"epoch": 10, "pop_size": 20},
}

# Initialize the comparator
compartor = MhaMlpComparator(
    optim_dict=optim_dict,
    task="regression",
    hidden_layers=(10, ),
    act_names="ELU",
    dropout_rates=None,
    act_output=None,
    obj_name="R2",
    verbose=True,
    seed=42,
)

### Perform comparison
# results = compartor.compare_cross_val_score(X_train, y_train, metric="RMSE", cv=4, n_trials=2, to_csv=True)
# print(results)

# results = compartor.compare_cross_validate(X_train, y_train, metrics=["MSE", "MAPE", "R2", "KGE", "NSE"],
#                                            cv=4, return_train_score=True, n_trials=2, to_csv=True)
# print(results)

results = compartor.compare_train_test(X_train, y_train, X_test, y_test,
                                       metrics=["MSE", "MAPE", "R2", "KGE", "NSE"], n_trials=2, to_csv=True)
print(results)
