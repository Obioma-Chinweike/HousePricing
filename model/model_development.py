# model_development.py
# House Price Prediction System - Part A
# Algorithm: Random Forest Regressor

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------
# Kaggle House Prices dataset (train.csv)
df = pd.read_csv("train.csv")

# -------------------------------------------------
# 2. Feature Selection
# Only 6 features selected from the approved list
# -------------------------------------------------
features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "Neighborhood"
]

target = "SalePrice"

df = df[features + [target]]

# -------------------------------------------------
# 3. Data Preprocessing
# -------------------------------------------------

# Handle missing values
# Numeric features → fill with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical feature → fill with mode
df["Neighborhood"] = df["Neighborhood"].fillna(df["Neighborhood"].mode()[0])

# Encode categorical variable (Neighborhood)
df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

# Split features and target
X = df.drop(target, axis=1)
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# 4. Model Implementation
# Random Forest does NOT require feature scaling
# -------------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# -------------------------------------------------
# 5. Model Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics")
print("------------------------")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")

# -------------------------------------------------
# 6. Save Trained Model
# -------------------------------------------------
joblib.dump(model, "house_price_model.pkl")
print("\nModel saved as house_price_model.pkl")

# -------------------------------------------------
# 7. Reload Model (Verification)
# -------------------------------------------------
loaded_model = joblib.load("house_price_model.pkl")
test_prediction = loaded_model.predict(X_test.iloc[:1])

print("\nModel reload verification successful.")
print("Sample prediction:", test_prediction[0])
