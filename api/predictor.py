from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "rf_week.pkl"
ENCODER_PATH = BASE_DIR / "encoder_week.pkl"
SCALER_PATH = BASE_DIR / "scaler_week.pkl"
FEATURE_COLS_PATH = BASE_DIR / "feature_cols_week.pkl"

categorical_cols = ["Product_Code", "Warehouse", "Product_Category"]

numeric_cols = [
    "Year", "Month", "WeekOfYear",
    "lag_1", "lag_2", "lag_4",
    "rolling_mean_4", "rolling_std_4"
]


def artifacts_status():
    return {
        "model": MODEL_PATH.exists(),
        "encoder": ENCODER_PATH.exists(),
        "scaler": SCALER_PATH.exists(),
        "feature_columns": FEATURE_COLS_PATH.exists()
    }


def load_artifacts():
    missing_files = []

    for path in [MODEL_PATH, ENCODER_PATH, SCALER_PATH, FEATURE_COLS_PATH]:
        if not path.exists():
            missing_files.append(path.name)

    if missing_files:
        raise FileNotFoundError(f"Missing model artifacts: {missing_files}")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    return model, encoder, scaler, feature_cols


def predict_demand(data: dict) -> float:
    model, encoder, scaler, feature_cols = load_artifacts()

    input_df = pd.DataFrame([{
        "Product_Code": data["Product_Code"],
        "Warehouse": data["Warehouse"],
        "Product_Category": data["Product_Category"],
        "Year": data["Year"],
        "Month": data["Month"],
        "WeekOfYear": data["WeekOfYear"],
        "lag_1": data["lag_1"],
        "lag_2": data["lag_2"],
        "lag_4": data["lag_4"],
        "rolling_mean_4": data["rolling_mean_4"],
        "rolling_std_4": data["rolling_std_4"]
    }])

    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    pred_log = model.predict(input_df[feature_cols])
    prediction = np.expm1(pred_log[0])

    return float(prediction)