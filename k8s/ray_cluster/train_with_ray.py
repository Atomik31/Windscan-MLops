import argparse
import pandas as pd
import time
import os
import boto3
import mlflow
import ray
import joblib
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from ray.util.joblib import register_ray

load_dotenv()

# Configuration MLflow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("turbine_maintenance_ray_distributed")

# Initialisation de Ray
# "auto" si tu es sur un cluster, sinon ray.init() pour du local parallélisé
ray.init(ignore_reinit_error=True)


def load_data_from_s3():
    """Télécharge le dataset d'entraînement depuis S3 — source unique de vérité."""
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "eu-north-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    s3_key = "data/dataset_train.csv"
    local_path = "/tmp/dataset_train.csv"

    print(f"Téléchargement du dataset depuis s3://{os.getenv('BUCKET_NAME')}/{s3_key}...")
    s3.download_file(Bucket=os.getenv("BUCKET_NAME"), Key=s3_key, Filename=local_path)
    return pd.read_csv(local_path, index_col=0)


if __name__ == "__main__":
    # 1. Parsing des arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    # 2. Chargement des données depuis S3
    df = load_data_from_s3()

    # Remove Split column if present (used for manual train/test split)
    if "Split" in df.columns:
        df = df.drop(columns=["Split"])

    # Use "Target" as target variable (binary: 0 or 1)
    target_col_name = "Target"
    X = df.drop(columns=[target_col_name])
    y = df[target_col_name]

    # Remove identifier columns (not useful for prediction)
    id_columns = ["Hour_Index", "Turbine_ID", "Split"]
    X = X.drop(columns=[col for col in id_columns if col in X.columns])

    # 3. Définition du Preprocessing (identique à ton original)
    categorical_features = X.select_dtypes("object").columns
    numerical_features = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                categorical_features,
            ),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    # 4. Pipeline & GridSearch
    # On définit une grille d'hyperparamètres basée sur les arguments reçus
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
    )

    param_grid = {
        "classifier__n_estimators": [args.n_estimators],
        "classifier__min_samples_split": [args.min_samples_split],
    }

    # CV=3 pour paralléliser l'entraînement des folds sur Ray
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)

    # 5. Entraînement distribué avec Ray
    print("🚀 Démarrage de l'entraînement distribué sur Ray...")
    register_ray()

    with joblib.parallel_backend("ray"):
        with mlflow.start_run() as run:
            start_time = time.time()

            # Autolog capture les paramètres et metrics de sklearn
            mlflow.sklearn.autolog(log_models=False)

            grid_search.fit(X, y)

            duration = time.time() - start_time
            print(f"✅ Entraînement terminé en {duration:.2f} secondes")

            # 6. Logging du modèle et Signature
            input_example = X.iloc[:5]
            predictions_example = grid_search.best_estimator_.predict(input_example)
            signature = infer_signature(input_example, predictions_example)

            mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                artifact_path="turbine_maintenance_predictor",
                registered_model_name="turbine_maintenance_predictor",
                signature=signature,
                input_example=input_example,
            )

            # 7. Gestion du Registre et de l'Alias
            client = MlflowClient()
            model_name = "turbine_maintenance_predictor"
            latest_versions = client.get_latest_versions(model_name, stages=["None"])

            if latest_versions:
                model_version = latest_versions[-1].version
                client.set_registered_model_alias(
                    name=model_name, alias="production", version=model_version
                )
                print(f"[INFO] Modèle version {model_version} marqué comme 'production'")

    print("...Done!")
