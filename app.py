from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

web_app = Flask(__name__)

# Load trained model
price_model = joblib.load("model/house_price_model.pkl")

# Feature columns used during training
trained_features = price_model.feature_names_in_

@web_app.route("/", methods=["GET", "POST"])
def index():
    estimated_price = None

    if request.method == "POST":
        user_features = {
            "OverallQual": int(request.form["OverallQual"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
            "GarageCars": int(request.form["GarageCars"]),
            "FullBath": int(request.form["FullBath"]),
            "Neighborhood": request.form["Neighborhood"]
        }

        features_df = pd.DataFrame([user_features])
        features_df = pd.get_dummies(features_df, columns=["Neighborhood"])
        features_df = features_df.reindex(columns=trained_features, fill_value=0)

        estimated_price = price_model.predict(features_df)[0]

    return render_template("index.html", prediction=estimated_price)


# Run App (Render Compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port, debug=False)
