"""
One-shot : injecte les 100 dernieres heures de donnees (Turbine 1) dans
wind_turbine_sensors et wind_turbine_predictions (Neon DB), pour simuler
un historique recent du pipeline.

Source : s3://projet-final-dsl-ft37-julien-charlier/data/wind_turbine_maintenance_test_data.csv
         (Turbine_ID == 1, 100 dernieres lignes)

Pour chaque ligne, dans l'ordre chronologique :
  1. INSERT dans wind_turbine_sensors (donnees nettoyees)
  2. POST /predict sur l'API de serving (turbine_maintenance_predictor@production)
  3. INSERT dans wind_turbine_predictions (capteurs + prediction)

Les created_at sont espaces d'1 heure et se terminent a l'heure actuelle (UTC),
pour simuler les 100 dernieres heures.

Variables d'environnement requises :
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   (lecture S3)
  PGHOST, PGPASSWORD                          (Neon DB)
  PGDATABASE, PGUSER, PGSSLMODE               (optionnels, defauts Neon)
  WINDSCAN_MODEL_API_URL                      (defaut: https://atomik31-model-serv-api.hf.space/predict)

Usage :
  python scripts/seed_last_100_hours.py
"""

import os
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
import psycopg2
import requests

S3_BUCKET = "projet-final-dsl-ft37-julien-charlier"
S3_KEY = "data/wind_turbine_maintenance_test_data.csv"
N_ROWS = 100
PREDICT_URL = os.getenv("WINDSCAN_MODEL_API_URL", "https://atomik31-model-serv-api.hf.space/predict")

# Mapping colonnes Neon DB (lowercase) -> champs attendus par l'API (memes que predict_and_store.py)
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


def main():
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "eu-north-1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    df = pd.read_csv(obj["Body"])
    df = df[df["Turbine_ID"] == 1].reset_index(drop=True)
    df.columns = df.columns.str.lower()
    df_tail = df.tail(N_ROWS).reset_index(drop=True)

    now = datetime.now(timezone.utc)
    timestamps = [now - timedelta(hours=(len(df_tail) - 1 - i)) for i in range(len(df_tail))]

    conn = psycopg2.connect(
        host=os.environ["PGHOST"],
        dbname=os.getenv("PGDATABASE", "neondb"),
        user=os.getenv("PGUSER", "neondb_owner"),
        password=os.environ["PGPASSWORD"],
        sslmode=os.getenv("PGSSLMODE", "require"),
    )

    inserted, predicted = 0, 0
    with conn:
        for ts, (_, row) in zip(timestamps, df_tail.iterrows()):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO wind_turbine_sensors (
                        turbine_id, rotor_speed_rpm, wind_speed_mps, power_output_kw,
                        gearbox_oil_temp_c, generator_bearing_temp_c, vibration_level_mmps,
                        ambient_temp_c, humidity_pct, maintenance_label, created_at
                    ) VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        float(row["rotor_speed_rpm"]), float(row["wind_speed_mps"]), float(row["power_output_kw"]),
                        float(row["gearbox_oil_temp_c"]), float(row["generator_bearing_temp_c"]), float(row["vibration_level_mmps"]),
                        float(row["ambient_temp_c"]), float(row["humidity_pct"]), int(row["maintenance_label"]), ts,
                    ),
                )
                sensor_id = cur.fetchone()[0]
            inserted += 1

            payload = {api_name: float(row[db_col]) for db_col, api_name in COLUMN_MAP.items()}
            payload["Maintenance_Label"] = int(row["maintenance_label"])

            try:
                resp = requests.post(PREDICT_URL, json=payload, timeout=60)
                resp.raise_for_status()
                prediction = resp.json().get("prediction")
            except Exception as e:
                print(f"[WARN] Echec /predict pour sensor_id={sensor_id} : {e}")
                continue

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO wind_turbine_predictions (
                        sensor_id, turbine_id, rotor_speed_rpm, wind_speed_mps, power_output_kw,
                        gearbox_oil_temp_c, generator_bearing_temp_c, vibration_level_mmps,
                        ambient_temp_c, humidity_pct, maintenance_label, prediction, created_at
                    ) VALUES (%s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        sensor_id,
                        float(row["rotor_speed_rpm"]), float(row["wind_speed_mps"]), float(row["power_output_kw"]),
                        float(row["gearbox_oil_temp_c"]), float(row["generator_bearing_temp_c"]), float(row["vibration_level_mmps"]),
                        float(row["ambient_temp_c"]), float(row["humidity_pct"]), int(row["maintenance_label"]),
                        int(prediction) if prediction is not None else None, ts,
                    ),
                )
            predicted += 1

            print(f"[OK] sensor_id={sensor_id} ts={ts.isoformat()} prediction={prediction}")

    conn.close()
    print(f"Termine. {inserted} lignes inserees dans wind_turbine_sensors, {predicted} predictions stockees.")


if __name__ == "__main__":
    main()
