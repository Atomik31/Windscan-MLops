import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable


EXPECTED_COLUMNS = [
    "turbine_id", "rotor_speed_rpm", "wind_speed_mps",
    "power_output_kw", "gearbox_oil_temp_c", "generator_bearing_temp_c",
    "vibration_level_mmps", "ambient_temp_c", "humidity_pct", "maintenance_label",
]

RANGE_CHECKS = {
    "rotor_speed_rpm":          (0, 30),
    "wind_speed_mps":           (0, 40),
    "power_output_kw":          (0, 5000),
    "gearbox_oil_temp_c":       (-20, 150),
    "generator_bearing_temp_c": (-20, 200),
    "vibration_level_mmps":     (0, 50),
    "ambient_temp_c":           (-40, 60),
    "humidity_pct":             (0, 100),
}


def validate_and_load_sensors(**context):
    ti = context["task_instance"]
    bucket = Variable.get("S3BucketName")
    raw_s3_key = ti.xcom_pull(task_ids="extract_raw_turbine_batch", key="data_raw_turbine_key")

    # Téléchargement depuis S3
    s3_hook = S3Hook(aws_conn_id="aws_default")
    local_path = s3_hook.download_file(key=raw_s3_key, bucket_name=bucket, local_path="/tmp")
    df = pd.read_csv(local_path)
    df.columns = df.columns.str.lower()

    print(f"[INFO] Données brutes reçues : {df.shape[0]} ligne(s), colonnes : {list(df.columns)}")

    # Validation du schéma
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"[SCHEMA ERROR] Colonnes manquantes : {missing}")
    print("[OK] Schéma valide — toutes les colonnes attendues sont présentes")

    # Validation des valeurs nulles
    nulls = df[EXPECTED_COLUMNS].isnull().sum()
    if nulls.any():
        raise ValueError(f"[QUALITY ERROR] Valeurs nulles détectées : {nulls[nulls > 0].to_dict()}")
    print("[OK] Aucune valeur nulle détectée")

    # Validation des plages — les lignes hors plage sont supprimées (outliers)
    mask_valid = pd.Series([True] * len(df), index=df.index)
    for col, (min_val, max_val) in RANGE_CHECKS.items():
        if col in df.columns:
            out = (df[col] < min_val) | (df[col] > max_val)
            if out.any():
                print(f"[OUTLIER] {col} : {out.sum()} ligne(s) hors plage [{min_val}, {max_val}] — supprimées")
                mask_valid &= ~out
            else:
                print(f"[OK] {col} dans la plage [{min_val}, {max_val}]")

    df = df[mask_valid]
    if df.empty:
        raise ValueError("[ERROR] Toutes les lignes ont été supprimées (outliers) — aucune donnée valide à charger.")

    print(f"[INFO] {len(df)} ligne(s) valide(s) après nettoyage outliers")

    df_to_load = df[EXPECTED_COLUMNS].copy()

    # Insertion dans wind_turbine_sensors
    postgres_hook = PostgresHook(postgres_conn_id="postgres_default")
    engine = postgres_hook.get_sqlalchemy_engine()
    df_to_load.to_sql("wind_turbine_sensors", engine, if_exists="append", index=False)

    print(f"[INFO] {len(df_to_load)} ligne(s) insérée(s) dans wind_turbine_sensors")
