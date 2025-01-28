import os
import joblib

import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error 


training_data_directory = "/opt/ml/input/data/train"
test_data_directory = "/opt/ml/input/data/test"

train_features_data = os.path.join(training_data_directory, "train_features.csv")
train_labels_data = os.path.join(training_data_directory, "train_labels.csv")

test_features_data = os.path.join(test_data_directory, "test_features.csv")
test_labels_data = os.path.join(test_data_directory, "test_labels.csv")

X_train = pd.read_csv(train_features_data, header=None)
y_train = pd.read_csv(train_labels_data, header=None)

model_dt = DecisionTreeRegressor()

model_dt.fit(X_train, y_train)

X_test = pd.read_csv(test_features_data, header=None)
y_test = pd.read_csv(test_labels_data, header=None)

y_pred = model_dt.predict(X_test)

print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")

model_output_directory = os.path.join("/opt/ml/model", "model.joblib")

print(f"Saving model to {model_output_directory}")
joblib.dump(model_dt, model_output_directory)