# app.py
# House Price Prediction Web Application (Flask)

from flask import Flask, render_template, request
import joblib
import pandas as pd

web_app = Flask(__name__)

# -------------------------------------------------
# Load trained model (NO retraining)
# -------------------------------------------------
price_model = joblib.load("model/house_price_model.pkl")

# Feature columns used during training
# NOTE: Must match model training exactly
trained_features = price_model.feature_names_in_

# -------------------------------------------------
# Routes
# -------------------------------------------------
@web_app.route("/", methods=["GET", "POST"])
def index():
    estimated_price = None

    if request.method == "POST":
        # Collect user input
        user_features = {
            "OverallQual": int(request.form["OverallQual"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
            "GarageCars": int(request.form["GarageCars"]),
            "FullBath": int(request.form["FullBath"]),
            "Neighborhood": request.form["Neighborhood"]
        }

        # Convert input to DataFrame
        features_df = pd.DataFrame([user_features])

        # Encode categorical variable (Neighborhood)
        features_df = pd.get_dummies(features_df, columns=["Neighborhood"])

        # Align columns with training data
        features_df = features_df.reindex(columns=trained_features, fill_value=0)

        # Make prediction
        estimated_price = price_model.predict(features_df)[0]

    return render_template("index.html", prediction=estimated_price)


# -------------------------------------------------
# Run App
# -------------------------------------------------
if __name__ == "__main__":
    web_app.run(debug=True)
