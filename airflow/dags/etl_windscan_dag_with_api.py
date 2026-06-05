from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator

from tasks_with_api.extract_windscan import extract_dataset_batch_to_s3
from tasks_with_api.validate_load_sensors import validate_and_load_sensors
from tasks_with_api.predict_and_store import predict_and_store


default_args = {
    "owner": "Julien.CHARLIER",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="etl_WINDSCAN_dag",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["demo", "etl", "windscan"],
) as dag:

    # =========================
    # 0) Création des tables si elles n'existent pas
    # =========================
    create_tables = PostgresOperator(
        task_id="create_tables",
        postgres_conn_id="postgres_default",
        sql="""
        CREATE TABLE IF NOT EXISTS wind_turbine_sensors (
            id                       SERIAL PRIMARY KEY,
            turbine_id               INTEGER,
            rotor_speed_rpm          NUMERIC,
            wind_speed_mps           NUMERIC,
            power_output_kw          NUMERIC,
            gearbox_oil_temp_c       NUMERIC,
            generator_bearing_temp_c NUMERIC,
            vibration_level_mmps     NUMERIC,
            ambient_temp_c           NUMERIC,
            humidity_pct             NUMERIC,
            maintenance_label        INTEGER,
            created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS wind_turbine_predictions (
            id                       SERIAL PRIMARY KEY,
            sensor_id                INTEGER REFERENCES wind_turbine_sensors(id),
            turbine_id               INTEGER,
            rotor_speed_rpm          NUMERIC,
            wind_speed_mps           NUMERIC,
            power_output_kw          NUMERIC,
            gearbox_oil_temp_c       NUMERIC,
            generator_bearing_temp_c NUMERIC,
            vibration_level_mmps     NUMERIC,
            ambient_temp_c           NUMERIC,
            humidity_pct             NUMERIC,
            maintenance_label        INTEGER,
            prediction               INTEGER,
            created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
    )

    # =========================
    # 1) Extract — dernière ligne du dataset S3
    # =========================
    extract_task = PythonOperator(
        task_id="extract_raw_turbine_batch",
        python_callable=extract_dataset_batch_to_s3,
        provide_context=True,
    )

    # =========================
    # 2) Validate + Load — schéma, qualité, insertion dans wind_turbine_sensors
    # =========================
    validate_load = PythonOperator(
        task_id="validate_and_load_sensors",
        python_callable=validate_and_load_sensors,
        provide_context=True,
    )

    # =========================
    # 3) Predict + Store — lit Neon DB, appelle /predict, stocke dans wind_turbine_predictions
    # =========================
    predict_store = PythonOperator(
        task_id="predict_and_store",
        python_callable=predict_and_store,
        provide_context=True,
    )

    create_tables >> extract_task >> validate_load >> predict_store
