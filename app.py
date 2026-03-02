from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("engine_model.pkl")

# Define feature order (MUST match training)
FEATURE_COLUMNS = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp"
]

# Safe operating ranges (define based on dataset or engineering assumptions)
safe_ranges = {
    "engine_rpm": (800, 5000),
    "lub_oil_pressure": (20, 60),
    "fuel_pressure": (40, 100),
    "coolant_pressure": (20, 50),
    "lub_oil_temp": (70, 120),
    "coolant_temp": (70, 110),
}

# Extract feature importance from model
importances = model.feature_importances_
importances_df = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)

@app.get("/")
def home():
    return {"message": "Engine Health API is running"}

@app.post("/predict")
def predict_engine_health(data: dict):

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])

    # Ensure correct column order
    input_df = input_df[FEATURE_COLUMNS]

    # --- Prediction ---
    probabilities = model.predict_proba(input_df)
    failure_probability = float(probabilities[0, 1])

    # Health classification
    if failure_probability < 0.3:
        engine_health = "GOOD"
    elif failure_probability <= 0.7:
        engine_health = "WARNING"
    else:
        engine_health = "CRITICAL"

    diagnostics = []
    current_readings = input_df.iloc[0]

    # --- Out-of-range check ---
    out_of_range_params = []
    for param, (lower, upper) in safe_ranges.items():
        value = current_readings[param]
        if not (lower <= value <= upper):
            status = "low" if value < lower else "high"
            out_of_range_params.append(
                f"{param} ({value}) is {status} (safe: {lower}-{upper})"
            )

    if out_of_range_params:
        diagnostics.append("Out-of-range parameters: " + "; ".join(out_of_range_params))

    # --- Top Important Features ---
    top_important_features = importances_df.head(3).index.tolist()

    potential_issues = []

    # Overheating
    if current_readings["coolant_temp"] > safe_ranges["coolant_temp"][1]:
        potential_issues.append("Possible OVERHEATING detected due to high coolant temperature.")

    # Lubrication
    if current_readings["lub_oil_pressure"] < safe_ranges["lub_oil_pressure"][0]:
        potential_issues.append("Possible LUBRICATION FAILURE due to low oil pressure.")

    # Pressure imbalance
    if (
        current_readings["fuel_pressure"] < safe_ranges["fuel_pressure"][0]
        or current_readings["fuel_pressure"] > safe_ranges["fuel_pressure"][1]
        or current_readings["coolant_pressure"] < safe_ranges["coolant_pressure"][0]
        or current_readings["coolant_pressure"] > safe_ranges["coolant_pressure"][1]
    ):
        potential_issues.append("Possible PRESSURE IMBALANCE detected.")

    if potential_issues:
        diagnostics.append("Potential issues: " + "; ".join(potential_issues))

    if not diagnostics:
        final_diagnostic = "All parameters within safe operating range."
    else:
        final_diagnostic = " ".join(diagnostics)

    return {
        "failure_probability": round(failure_probability, 4),
        "engine_health": engine_health,
        "diagnostic_analysis": final_diagnostic,
        "top_influential_features": top_important_features
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)