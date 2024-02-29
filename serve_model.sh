#!/usr/bin/env sh

# Deploying a Model using the mlflow cli
export MLFLOW_TRACKING_URI="sqlite:///mydb.sqlite"

# Serve the model from the 
mlflow models serve -m "models:/StartupModels/1" --no-conda