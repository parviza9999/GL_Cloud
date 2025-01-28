import os
import json
import joblib
import tarfile

import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score


model_path = f"/opt/ml/processing/model/model.tar.gz"

with tarfile.open(model_path) as tar:
     tar.extractall(path=".")

model = joblib.load('model.joblib')

print("Loading test input data")
test_data_directory = "/opt/ml/processing/test"
test_features_data = os.path.join(test_data_directory, "test_features.csv")
test_labels_data = os.path.join(test_data_directory, "test_labels.csv")

X_test = pd.read_csv(test_features_data, header=None)
y_test = pd.read_csv(test_labels_data, header=None)

y_pred = model.predict(X_test)

print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)};")

report_dict = {
        "regression_metrics": {
                "mse": {
                        "value": mean_squared_error(y_test, y_pred)
                },
                "rmse": {
                        "value": mean_squared_error(y_test, y_pred, squared=False)
                },
                "r2": {
                        "value": r2_score(y_test, y_pred)
                }
        }
}

evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")

with open(evaluation_output_path, "w") as f:
      f.write(json.dumps(report_dict))