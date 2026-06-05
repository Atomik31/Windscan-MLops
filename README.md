# Maintenance Predictive des Eoliennes — Pipeline MLOps complet

Projet final de la certification AIA (Architecte en Intelligence Artificielle — Jedha).

---

## Contexte

WindScan, opérateur de parcs éoliens, veut anticiper les pannes de turbines avant qu'elles surviennent. Les turbines envoient en continu des mesures capteurs (vitesse rotor, température boîte de vitesses, vibrations...). L'enjeu est de construire un pipeline complet : de l'entraînement distribué du modèle jusqu'à l'inférence automatisée en batch sur les nouvelles données.

---

## Ce que j'ai fait

**Entraînement distribué (Ray sur Kubernetes)**

Un Random Forest entraîné avec `GridSearchCV` (cv=3) sur les données capteurs des turbines. L'entraînement est distribué sur un cluster Ray déployé via l'opérateur KubeRay sur Minikube. `joblib` avec le backend `ray` distribue automatiquement les folds de cross-validation sur les workers.

Le modèle est enregistré dans MLflow sous `turbine_maintenance_predictor` avec l'alias `staging`.

**Serving — FastAPI sur Hugging Face Spaces**

Une API FastAPI charge le modèle depuis MLflow au démarrage et expose deux endpoints : `/health` et `/predict`. Elle tourne dans un conteneur Docker déployé sur Hugging Face Spaces.

**Pipeline d'inférence — DAG Airflow**

Un DAG se déclenche manuellement et enchaîne 3 tâches :

```
┌─────────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐     ┌─────────────┐
│     1. Extract      │────▶│      2. Predict       │────▶│       3. Load         │────▶│   Neon DB   │
│ extrait la dernière │     │ appelle l'API /predict │     │ XCom → PostgreSQL     │     │ (PostgreSQL)│
│ ligne du dataset S3 │     │ ligne par ligne        │     │ wind_turbine_         │     │             │
│                     │     │ predictions → XCom     │     │ predictions           │     │             │
└─────────────────────┘     └──────────┬───────────┘     └───────────────────────┘     └─────────────┘
                                       │ POST /predict
                                       ▼
                             atomik31-model-serv-api
                                  .hf.space
```

Les prédictions transitent entre les tâches via XCom (JSON) et sont insérées directement dans Neon DB.

---

## Stack

- Python — scikit-learn, pandas, FastAPI, MLflow
- Ray 2.x + KubeRay (entraînement distribué)
- Kubernetes — Minikube (cluster local)
- Apache Airflow 2.10 (Docker Compose, LocalExecutor)
- MLflow Model Registry (Hugging Face Spaces)
- AWS S3 (données brutes + prédictions CSV en transit)
- Neon DB (PostgreSQL managé — stockage final des prédictions)
- Hugging Face Spaces (MLflow server + API de serving)

---

## Architecture

```
                    ENTRAINEMENT
 ┌───────────────────────────────────────────────────┐
 │  Kubernetes (Minikube)                            │
 │  ┌─────────────┐   ┌─────────────┐               │
 │  │  Ray Head   │◀─▶│ Ray Workers │               │
 │  └──────┬──────┘   └─────────────┘               │
 │         │  train_with_ray.py                      │
 │         │  (RF + GridSearchCV, cv=3 folds)        │
 └─────────┼─────────────────────────────────────────┘
           │ log model
           ▼
   ┌───────────────────────┐     load at startup     ┌──────────────────────────┐
   │  MLflow (HF Spaces)   │────────────────────────▶│  FastAPI (HF Spaces)     │
   │  turbine_maintenance  │                          │  /health  /predict        │
   │  _predictor  v5       │                          └──────────────────────────┘
   └───────────────────────┘                                      ▲
                                                                  │ POST /predict
                    INFERENCE BATCH                               │
 ┌────────────────────────────────────────────────────────────────┼──────┐
 │  Apache Airflow — etl_WINDSCAN_dag                             │      │
 │                                                                │      │
 │  [Extract]──▶[Predict]──────────────────────────────────────────      │
 │               └──▶ [Load] ──▶ Neon DB (wind_turbine_predictions)      │
 └───────────────────────────────────────────────────────────────────────┘
```

---

## Structure

```
Projet-final/
├── docs/
│   └── project_overview_final.md   # Enonce du projet
├── k8s/
│   └── ray_cluster/
│       ├── train_with_ray.py        # Script d'entrainement distribue
│       ├── ray_cluster.yaml         # Helm values KubeRay
│       ├── runtime.yaml             # Env Ray (dependances pip)
│       ├── dataset_train.csv        # Dataset d'entraînement turbines
│       └── requirements.txt
├── modelservedapi/
│   ├── app.py                       # API FastAPI (/health + /predict)
│   ├── Dockerfile
│   └── requirements.txt
├── mlflowfinalproject/
│   └── Dockerfile                   # MLflow server sur HF Spaces
├── airflow/
│   ├── dags/
│   │   ├── etl_WINDSCAN_dag_with_api.py       # DAG principal (turbines)
│   │   ├── etl_attrition_dag_with_pkl.py      # DAG secondaire (IBM)
│   │   └── tasks_with_api/
│   │       ├── extract_windscan.py             # Tache 1 : extraction S3
│   │       ├── transform_predict_windscan.py   # Tache 2 : appel API + predictions
│   │       └── load_to_postgres.py             # Tache 3 : XCom → Neon DB
│   ├── docker-compose.yaml
│   └── Dockerfile
├── Deployment.md                    # Commandes de deploiement pas a pas
└── README.md
```

---

## Lancer le projet

Voir [Deployment.md](Deployment.md) pour le detail complet. En resume :

```bash
# 1. Demarrer le cluster K8s
minikube start --driver=docker

# 2. Port-forward le dashboard Ray
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265

# 3. Soumettre l'entrainement
ray job submit --runtime-env=runtime.yaml --address="http://127.0.0.1:8265" -- python train_with_ray.py

# 4. Lancer Airflow
cd airflow && docker-compose up -d
# Interface : http://localhost:8081
```

---

Julien CHARLIER — [(Github : Atomik31)](https://github.com/Atomik31)
