import json
import requests
import pandas as pd
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable


# Mapping colonnes Neon DB (lowercase) → champs attendus par l'API
# Uniquement les features sur lesquelles le modèle a été entraîné
# (Hour_Index et Turbine_ID ont été exclus à l'entraînement)
COLUMN_MAP = {
    "rotor_speed_rpm":          "Rotor_Speed_RPM",
    "wind_speed_mps":           "Wind_Speed_mps",
    "power_output_kw":          "Power_Output_kW",
    "gearbox_oil_temp_c":       "Gearbox_Oil_Temp_C",
    "generator_bearing_temp_c": "Generator_Bearing_Temp_C",
    "vibration_level_mmps":     "Vibration_Level_mmps",
    "ambient_temp_c":           "Ambient_Temp_C",
    "humidity_pct":             "Humidity_pct",
    "maintenance_label":        "Maintenance_Label",
}


def predict_and_store(**context):
    ti = context["task_instance"]

    model_api_base_url = Variable.get("WINDSCAN_MODEL_API_BASE_URL")
    model_api_predict_endpoint = Variable.get(
        "WINDSCAN_MODEL_API_PREDICT_ENDPOINT", default_var="/predict"
    )
    request_timeout = int(Variable.get("WINDSCAN_MODEL_API_TIMEOUT", default_var="120"))

    if model_api_predict_endpoint.startswith(("http://", "https://")):
        predict_url = model_api_predict_endpoint
    else:
        predict_url = f"{model_api_base_url.rstrip('/')}{model_api_predict_endpoint}"

    # Lecture de la dernière ligne depuis wind_turbine_sensors
    postgres_hook = PostgresHook(postgres_conn_id="postgres_default")
    engine = postgres_hook.get_sqlalchemy_engine()
    df = pd.read_sql(
        "SELECT * FROM wind_turbine_sensors ORDER BY id DESC LIMIT 1",
        engine,
    )

    if df.empty:
        raise ValueError("[ERROR] Aucune donnée dans wind_turbine_sensors.")

    sensor_id = int(df["id"].iloc[0])
    row = df.iloc[0]

    # Construction du payload API (reconstitution de la casse originale)
    payload = {api_name: row[db_col] for db_col, api_name in COLUMN_MAP.items()}
    payload = json.loads(json.dumps(payload, default=float))

    print(f"[INFO] Appel API : {predict_url}")
    print(f"[DEBUG] Payload : {payload}")

    response = requests.post(predict_url, json=payload, timeout=request_timeout)
    response.raise_for_status()

    prediction = response.json().get("prediction")
    if isinstance(prediction, list):
        prediction = prediction[0]

    print(f"[INFO] Prédiction reçue : {prediction} (sensor_id={sensor_id})")

    # Stockage dans wind_turbine_predictions — toutes les valeurs capteurs + prédiction
    pred_df = pd.DataFrame([{
        "sensor_id":                sensor_id,
        "turbine_id":               int(row["turbine_id"]),
        "rotor_speed_rpm":          float(row["rotor_speed_rpm"]),
        "wind_speed_mps":           float(row["wind_speed_mps"]),
        "power_output_kw":          float(row["power_output_kw"]),
        "gearbox_oil_temp_c":       float(row["gearbox_oil_temp_c"]),
        "generator_bearing_temp_c": float(row["generator_bearing_temp_c"]),
        "vibration_level_mmps":     float(row["vibration_level_mmps"]),
        "ambient_temp_c":           float(row["ambient_temp_c"]),
        "humidity_pct":             float(row["humidity_pct"]),
        "maintenance_label":        int(row["maintenance_label"]),
        "prediction":               int(prediction) if prediction is not None else None,
    }])
    pred_df.to_sql("wind_turbine_predictions", engine, if_exists="append", index=False)

    print(f"[INFO] Prédiction stockée dans wind_turbine_predictions (sensor_id={sensor_id})")
    ti.xcom_push(key="prediction", value=prediction)
