import json
import requests
import pandas as pd

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable


def predict_with_model_turbine(**context):
    """
    Read extracted raw turbine batch from S3,
    call model serving API for each row, push predictions via XCom.
    """
    ti = context["task_instance"]

    # =========================
    # 1) Read config from Airflow Variables
    # =========================
    model_api_base_url = Variable.get("WINDSCAN_MODEL_API_BASE_URL")
    model_api_predict_endpoint = Variable.get(
        "WINDSCAN_MODEL_API_PREDICT_ENDPOINT",
        default_var="/predict",
    )
    request_timeout = int(
        Variable.get("WINDSCAN_MODEL_API_TIMEOUT", default_var="120")
    )
    bucket = Variable.get("S3BucketName")

    # =========================
    # 2) Pull raw extract S3 key from XCom
    # =========================
    raw_s3_key = ti.xcom_pull(
        task_ids="extract_raw_turbine_batch",
        key="data_raw_turbine_key",
    )
    if not raw_s3_key:
        raise ValueError("Missing XCom raw_s3_key (key='data_raw_turbine_key').")

    # =========================
    # 3) Download raw batch from S3
    # =========================
    s3_hook = S3Hook(aws_conn_id="aws_default")
    local_raw_path = s3_hook.download_file(
        key=raw_s3_key,
        bucket_name=bucket,
        local_path="/tmp",
    )
    print(f"[INFO] Downloaded raw batch: s3://{bucket}/{raw_s3_key} -> {local_raw_path}")

    df = pd.read_csv(local_raw_path)

    # =========================
    # 4) Filter features (exclude Target and Split columns)
    # =========================
    feature_columns = [
        "Hour_Index",
        "Turbine_ID",
        "Rotor_Speed_RPM",
        "Wind_Speed_mps",
        "Power_Output_kW",
        "Gearbox_Oil_Temp_C",
        "Generator_Bearing_Temp_C",
        "Vibration_Level_mmps",
        "Ambient_Temp_C",
        "Humidity_pct",
        "Maintenance_Label",
    ]
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    features = df[feature_columns].copy()
    print(f"[INFO] Features dataframe prepared: shape={features.shape}")

    # =========================
    # 5) Call prediction API row by row
    # =========================
    if model_api_predict_endpoint.startswith(("http://", "https://")):
        predict_url = model_api_predict_endpoint
    else:
        predict_url = f"{model_api_base_url.rstrip('/')}{model_api_predict_endpoint}"

    predictions = []

    for idx, row in features.iterrows():
        payload = json.loads(row.to_json())
        try:
            print(f"[DEBUG] Row {idx} payload: {payload}")
            response = requests.post(predict_url, json=payload, timeout=request_timeout)
            response.raise_for_status()
            result = response.json()
            prediction = result.get("prediction")
            if isinstance(prediction, list) and len(prediction) > 0:
                prediction = prediction[0]
            predictions.append(prediction)
            print(f"[INFO] Prediction OK for row {idx}: {prediction}")
        except Exception as e:
            print(f"[ERROR] Prediction failed for row {idx}: {e}")
            print(f"[DEBUG] Response status: {getattr(e.response, 'status_code', 'N/A')}")
            print(f"[DEBUG] Response text: {getattr(e.response, 'text', 'N/A')}")
            predictions.append(None)

    # =========================
    # 6) Build result dataframe
    # =========================
    result_df = features.copy()
    result_df["prediction"] = predictions

    if "Target" in df.columns:
        result_df["target_actual"] = df["Target"].values

    # =========================
    # 7) Push predictions via XCom (pas de transit S3)
    # =========================
    ti.xcom_push(key="turbine_predictions_json", value=result_df.to_json(orient="records"))
    ti.xcom_push(key="turbine_predictions_count", value=len(result_df))
    print(f"[INFO] {len(result_df)} predictions pushed via XCom")
