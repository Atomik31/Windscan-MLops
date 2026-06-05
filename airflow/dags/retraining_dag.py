from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable


def submit_ray_training_job(**context):
    """
    Soumet un job de réentraînement au cluster Ray via son API Dashboard.
    Déclenché automatiquement par le DAG monitoring_WINDSCAN_dag en cas de drift.

    Prérequis : le port-forward Ray doit être actif sur le host :
      kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265

    Depuis Docker (Mac), le host est accessible via host.docker.internal.
    """
    ray_url = Variable.get("RAY_DASHBOARD_URL", default_var="http://host.docker.internal:8265")

    payload = {
        "entrypoint": "python train_with_ray.py",
        "runtime_env": {
            "working_dir": ".",
            "pip": [
                "scikit-learn==1.4.2",
                "mlflow",
                "pandas",
                "boto3",
                "python-dotenv",
            ],
        },
    }

    print(f"[INFO] Soumission du job de réentraînement à Ray : {ray_url}")
    response = requests.post(f"{ray_url}/api/jobs/", json=payload, timeout=30)
    response.raise_for_status()

    job_id = response.json().get("job_id")
    print(f"[INFO] Job Ray soumis avec succès : {job_id}")

    context["task_instance"].xcom_push(key="ray_job_id", value=job_id)
    return job_id


default_args = {
    "owner": "Julien.CHARLIER",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retraining_WINDSCAN_dag",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # déclenché uniquement par monitoring_WINDSCAN_dag
    catchup=False,
    tags=["retraining", "windscan", "ray"],
) as dag:

    retrain = PythonOperator(
        task_id="submit_ray_training_job",
        python_callable=submit_ray_training_job,
        provide_context=True,
    )
