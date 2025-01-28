import os
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split


input_data_path = os.path.join("/opt/ml/processing/input", "marketing3.csv")
data = pd.read_csv(input_data_path)

target = 'Sales'
numeric_features = ['TV','Radio','Newspaper']


X = data.drop(columns=[target])
y = data[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                test_size=0.2,
                                                random_state=42)

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features))


transformed_Xtrain = preprocessor.fit_transform(Xtrain)
transformed_Xtest = preprocessor.transform(Xtest)

train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

pd.DataFrame(transformed_Xtrain).to_csv(train_features_output_path, 
                                        header=False, index=False)
pd.DataFrame(transformed_Xtest).to_csv(test_features_output_path, 
                                       header=False, index=False)

ytrain.to_csv(train_labels_output_path, header=False, index=False)
ytest.to_csv(test_labels_output_path, header=False, index=False)