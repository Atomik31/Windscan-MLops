import pandas as pd
from airflow.hooks.postgres_hook import PostgresHook


def load_predictions_to_postgres(**context):
    ti = context["task_instance"]
    predictions_json = ti.xcom_pull(
        task_ids="predict_with_model_turbine",
        key="turbine_predictions_json",
    )
    if not predictions_json:
        raise ValueError("No predictions data found in XCom.")

    df = pd.read_json(predictions_json, orient="records")
    df.columns = df.columns.str.lower()

    postgres_hook = PostgresHook(postgres_conn_id="postgres_default")
    engine = postgres_hook.get_sqlalchemy_engine()
    df.to_sql("wind_turbine_predictions", engine, if_exists="append", index=False)
    print(f"[INFO] {len(df)} predictions inserted into wind_turbine_predictions")
