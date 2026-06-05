from datetime import datetime, timedelta

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable


FEATURE_COLUMNS = [
    "rotor_speed_rpm", "wind_speed_mps", "power_output_kw",
    "gearbox_oil_temp_c", "generator_bearing_temp_c", "vibration_level_mmps",
    "ambient_temp_c", "humidity_pct", "maintenance_label",
]

MIN_ROWS_FOR_DETECTION = 10  # nombre minimum de lignes pour évaluer le drift


def check_data_drift(**context):
    """
    Compare la distribution des données capteurs récentes (wind_turbine_sensors)
    au dataset d'entraînement (référence) via Evidently.
    Si drift détecté → déclenche le DAG de réentraînement.
    """
    # 1. Données de référence : dataset d'entraînement sur S3
    bucket = Variable.get("S3BucketName")
    s3_prefix = Variable.get("DATA_S3_PREFIX")
    s3_hook = S3Hook(aws_conn_id="aws_default")
    local_path = s3_hook.download_file(
        key=f"{s3_prefix}/dataset_train.csv",
        bucket_name=bucket,
        local_path="/tmp",
    )
    reference_df = pd.read_csv(local_path)
    reference_df.columns = reference_df.columns.str.lower()
    reference_df = reference_df[FEATURE_COLUMNS].dropna()
    print(f"[INFO] Référence chargée : {len(reference_df)} lignes")

    # 2. Données courantes : dernières lignes de wind_turbine_sensors (Neon DB)
    postgres_hook = PostgresHook(postgres_conn_id="postgres_default")
    engine = postgres_hook.get_sqlalchemy_engine()
    current_df = pd.read_sql(
        f"SELECT {', '.join(FEATURE_COLUMNS)} FROM wind_turbine_sensors ORDER BY id DESC LIMIT 100",
        engine,
    )
    print(f"[INFO] Données courantes : {len(current_df)} lignes")

    if len(current_df) < MIN_ROWS_FOR_DETECTION:
        print(f"[INFO] Pas assez de données ({len(current_df)} < {MIN_ROWS_FOR_DETECTION}) — drift non évalué")
        return "no_drift"

    # 3. Rapport Evidently
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()

    dataset_drift = result["metrics"][0]["result"]["dataset_drift"]
    share_drifted = result["metrics"][0]["result"]["share_of_drifted_columns"]

    print(f"[INFO] Drift détecté : {dataset_drift}")
    print(f"[INFO] Part de features en drift : {share_drifted:.0%}")

    context["task_instance"].xcom_push(key="dataset_drift", value=dataset_drift)
    context["task_instance"].xcom_push(key="share_drifted", value=share_drifted)

    return "trigger_retraining" if dataset_drift else "no_drift"


def log_no_drift(**context):
    print("[OK] Aucun drift détecté — modèle en production stable")


default_args = {
    "owner": "Julien.CHARLIER",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="monitoring_WINDSCAN_dag",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["monitoring", "windscan", "evidently"],
) as dag:

    drift_check = BranchPythonOperator(
        task_id="check_data_drift",
        python_callable=check_data_drift,
        provide_context=True,
    )

    no_drift = PythonOperator(
        task_id="no_drift",
        python_callable=log_no_drift,
        provide_context=True,
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="retraining_WINDSCAN_dag",
        wait_for_completion=False,
    )

    drift_check >> [no_drift, trigger_retraining]
