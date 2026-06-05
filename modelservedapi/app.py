import os
import mlflow
import pandas as pd
import uvicorn
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Literal, List, Union

# -----------------------------------------------------------------------------
# ENV + MLflow setup
# -----------------------------------------------------------------------------
# Sur Hugging Face, ces variables sont lues depuis les "Secrets"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS")

# On force l'URI pour mlflow
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def build_model_uri() -> str:
    if MODEL_ALIAS:
        return f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    return f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE or 'Production'}"
    # models:/ibm_attrition_detector@production


MODEL_URI = build_model_uri()
MODEL = None

# -----------------------------------------------------------------------------
# FastAPI Setup
# -----------------------------------------------------------------------------
app = FastAPI(title="🌬️ Wind Turbine Prediction API")


class PredictionFeatures(BaseModel):
    Rotor_Speed_RPM: float
    Wind_Speed_mps: float
    Power_Output_kW: float
    Gearbox_Oil_Temp_C: float
    Generator_Bearing_Temp_C: float
    Vibration_Level_mmps: float
    Ambient_Temp_C: float
    Humidity_pct: float
    Maintenance_Label: Union[int, float]


# -----------------------------------------------------------------------------
# Startup: CHARGEMENT BLOQUANT (Solution au bug 500)
# -----------------------------------------------------------------------------
@app.on_event("startup")
def load_model_sync():
    global MODEL
    print(f"🚀 [INFO] Attempting to load model: {MODEL_URI}")
    try:
        # On attend que le chargement soit fini avant de rendre l'API disponible
        MODEL = mlflow.sklearn.load_model(MODEL_URI)
        # models:/wind_turbine_predictor@production
        print("✅ [INFO] Model loaded successfully!")
    except Exception as e:
        print(f"❌ [ERROR] Failed to load model: {e}")
        # En cas d'échec, on laisse MODEL à None pour que /health le signale


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_loaded": MODEL is not None,
    }


@app.post("/predict")
async def predict(payload: PredictionFeatures):
    if MODEL is None:
        raise HTTPException(
            status_code=503, detail="Model is still loading or failed to load."
        )

    # Conversion pydantic -> dict -> DataFrame
    df = pd.DataFrame([payload.dict()])
    pred = MODEL.predict(df)
    return {"prediction": int(pred[0])}


if __name__ == "__main__":
    # Port 7860 est le standard pour Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))