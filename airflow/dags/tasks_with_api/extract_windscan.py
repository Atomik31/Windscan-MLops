import pandas as pd
from datetime import datetime

from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


def extract_dataset_batch_to_s3(**context):
    """
    Simule l'arrivée de données capteurs en temps réel :
    lit une ligne aléatoire du dataset de test (wind_turbine_maintenance_test_data.csv)
    stocké sur S3, upload le sample horodaté sur S3, pousse la clé via XCom.

    Le dataset de test est distinct du dataset d'entraînement (dataset_train.csv)
    pour simuler des données capteurs inconnues du modèle.
    """
    bucket = Variable.get("S3BucketName")
    s3_prefix = Variable.get("DATA_S3_PREFIX")
    s3_prefix_sample = Variable.get("SAMPLE_S3_PREFIX")

    # Dataset de test — séparé du dataset d'entraînement
    s3_key = f"{s3_prefix}/wind_turbine_maintenance_test_data.csv"
    s3_hook = S3Hook(aws_conn_id="aws_default")
    local_path = s3_hook.download_file(
        key=s3_key,
        bucket_name=bucket,
        local_path="/tmp",
    )
    df = pd.read_csv(local_path)

    # Tirage aléatoire d'une ligne — simule une nouvelle mesure capteur
    df_sample = df.sample(n=1, random_state=None)

    print(f"[INFO] Dataset de test chargé : {len(df)} lignes disponibles")
    print(f"[INFO] Ligne sélectionnée : index={df_sample.index[0]}")

    local_sample_path = "/tmp/sample.csv"
    df_sample.to_csv(local_sample_path, index=False)

    # Upload du sample horodaté sur S3 (donnée brute archivée)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_key_sample = f"{s3_prefix_sample}/sample_{ts}.csv"
    s3_hook.load_file(
        filename=local_sample_path,
        key=s3_key_sample,
        bucket_name=bucket,
        replace=True,
    )

    print(f"[INFO] Sample archivé sur S3 : s3://{bucket}/{s3_key_sample}")

    ti = context["task_instance"]
    ti.xcom_push(key="data_raw_turbine_key", value=s3_key_sample)
    ti.xcom_push(key="data_recorded_collected", value=len(df_sample))
