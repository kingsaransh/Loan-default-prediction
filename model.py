import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"), low_memory=False)

print("Columns:", df.columns)

# Target column
target = "loss"

# Features & target
X = df.drop(target, axis=1)
y = df[target]

# Drop ID column if exists
if "id" in X.columns:
    X = X.drop("id", axis=1)

# 🔥 Convert ALL columns to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# 🧹 Clean data
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# NOW safe to clip
X = X.clip(-1e6, 1e6)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_val)

# Evaluate
mse = mean_squared_error(y_val, pred)
print("MSE:", mse)

# Save
joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(X.columns, os.path.join(BASE_DIR, "columns.pkl"))

print("✅ Model saved successfully")