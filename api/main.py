from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.predictor import artifacts_status, predict_demand


app = FastAPI(
    title="Demand Forecasting API",
    description="REST API for weekly demand forecasting model.",
    version="1.0.0"
)


class PredictionInput(BaseModel):
    Product_Code: str = Field(..., example="Product_0979")
    Warehouse: str = Field(..., example="Whse_J")
    Product_Category: str = Field(..., example="Category_028")
    Year: int = Field(..., example=2017)
    Month: int = Field(..., ge=1, le=12, example=1)
    WeekOfYear: int = Field(..., ge=1, le=53, example=1)
    lag_1: float = Field(..., ge=0, example=500.0)
    lag_2: float = Field(..., ge=0, example=500.0)
    lag_4: float = Field(..., ge=0, example=500.0)
    rolling_mean_4: float = Field(..., ge=0, example=500.0)
    rolling_std_4: float = Field(..., ge=0, example=0.0)


@app.get("/")
def root():
    return {"message": "Demand Forecasting API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": "Random Forest weekly demand forecasting model",
        "artifacts": artifacts_status()
    }


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        prediction = predict_demand(input_data.model_dump())

        return {
            "predicted_weekly_demand": round(prediction, 2)
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(error)}")