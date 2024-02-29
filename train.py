import sys
import argparse
import mlflow
import mlflow.xgboost
import xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set MLflow tracking server and the Experiment Name
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")
mlflow.set_experiment("PotentialStartups")

# Parse out our parameter Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators')
parser.add_argument('--max_depth')
args = parser.parse_args()

# Set the values of our Arguments
n_estimators = int(args.n_estimators)
max_depth = int(args.max_depth)

# Set up the Training Data
# Load the Dataset
df = pd.read_csv('startups_profit.csv', index_col=False)
df['State']=df['State'].map({'New York':0,'Florida':1, 'California': 2}).astype(int)

# Training Data
X = df[["R&D Spend", "Administration", "Marketing Spend","State"]]
y = df[["Profit"]]
X, y = df.iloc[:, :-1], df.iloc[:, -1] 

# Setting up train test split
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), train_size=0.7,random_state=0)

# Start a training Run and autolog it.
with mlflow.start_run() as run:
    mlflow.xgboost.autolog()
    xgbr = xgboost.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth) 
    xgbr.fit(X_train, y_train)

    # Evaluate our Model using MLflow
    eval_data = X_test
    eval_data["Profits"] = y_test
    
    # Load the model
    model_uri = mlflow.get_artifact_uri("model")
    
    # Evaluate the model and autolog it
    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="Profits",
        model_type="regressor",
        evaluators="default"
    )
