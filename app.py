# app.py
# House Price Prediction Web Application (Flask)

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# -------------------------------------------------
# Load trained model (NO retraining)
# -------------------------------------------------
model = joblib.load("model/house_price_model.pkl")

# Feature columns used during training
# NOTE: Must match model training exactly
feature_columns = model.feature_names_in_

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        # Collect user input
        input_data = {
            "OverallQual": int(request.form["OverallQual"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
            "GarageCars": int(request.form["GarageCars"]),
            "FullBath": int(request.form["FullBath"]),
            "Neighborhood": request.form["Neighborhood"]
        }

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variable (Neighborhood)
        input_df = pd.get_dummies(input_df, columns=["Neighborhood"])

        # Align columns with training data
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]

    return render_template("index.html", prediction=prediction)


# -------------------------------------------------
# Run App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
