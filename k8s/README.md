# README : Déploiement de MLflow avec PostgreSQL et Ray sur Kubernetes

Ce guide explique comment déployer MLflow avec une base PostgreSQL et soumettre un job d’entraînement via Ray (KubeRay), le tout sur Minikube.

---

## Prérequis

- [Minikube](https://minikube.sigs.k8s.io/docs/start/) installé sur votre machine.
- Helm configuré sur votre environnement.
- Accès à un terminal.

---

## Étapes de déploiement

### 1. Initialisation de Minikube

1. Supprimez tout cluster existant pour partir d’un environnement propre :
   ```bash
   minikube delete
   ```
2. Lancez un nouveau cluster Minikube :
   ```bash
   minikube start --cpus=7 --memory=10000
   ```
3. (Optionnel) Accédez au tableau de bord Kubernetes dans une autre fenêtre terminal :
   ```bash
   minikube dashboard
   ```

---

### 2. Configuration de PostgreSQL avec Helm

1. Ajoutez le dépôt Helm Bitnami (si ce n’est pas encore fait) :
   ```bash
   helm repo add bitnami https://charts.bitnami.com/bitnami
   ```
2. Mettez à jour vos dépôts Helm :
   ```bash
   helm repo update
   ```
3. Déployer PostgreSQL
   ```bash
   helm install mehdipostgres bitnami/postgresql -f config.yaml
   ```

Exemple config.yaml :

```yaml
global:
  postgresql:
    auth:
      username: "mehdi"
      password: "rocket"
      database: "mehdidb"
```

---

### 3. Configuration de MLflow

1. Mettez à jour le fichier `mlflow-secrets.yaml` pour refléter l'URI de la base de données PostgreSQL :

   ```yaml
   BACKEND_STORE_URI: postgresql://mehdi:rocket@mehdipostgres-postgresql.default.svc.cluster.local:5432/mehdidb
   ```

   **Note :** L'URI doit suivre la syntaxe suivante :

   ```
   postgresql://<utilisateur>:<mot_de_passe>@<nom_release>-postgresql.default.svc.cluster.local:5432/<nom_base_de_données>
   ```

2. Appliquez les fichiers de configuration MLflow :

   ```bash
   kubectl apply -f mlflow-secrets.yaml
   kubectl apply -f mlflow-deployment.yaml
   kubectl apply -f mlflow-service.yaml
   ```

Accéder à l’interface MLflow :

```bash
minikube service mlflow-service
```

---

### 4. Déployer Ray (depuis le repo Ray)

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
```

```bash
helm repo update
```

```bash
helm install kuberay-operator kuberay/kuberay-operator --version 1.4.0
```

```bash
helm install raycluster kuberay/ray-cluster --version 1.4.0 -f ray-cluster.yaml
```

Exemple ray-cluster.yaml minimal :

```yaml
head:
  enableInTreeAutoscaling: true
  resources:
    limits:
      cpu: "4"
      # To avoid out-of-memory issues, never allocate less than 2G memory for the Ray head.
      memory: "4G"
    requests:
      cpu: "4"
      memory: "4G"

worker:
  replicas: 1
  resources:
    limits:
      cpu: "2"
      memory: "4G"
    requests:
      cpu: "2"
      memory: "4G"
```

---

### 5. Ouvrir le tableau de bord Ray (tout en exposant ray hors du cluster)

```bash
kubectl port-forward --address 0.0.0.0 service/raycluster-kuberay-head-svc 8265:8265
```

---

### 6. Soumettre un job d’entraînement

Dans un autre terminal :

1. Exemple runtime.yaml :

```yaml
---
working_dir: "./"
pip:
  - numpy
  - joblib
  - scikit-learn==1.4.2
  - mlflow==2.21.3
  - boto3
  - requests>=2.31.0,<3
env_vars:
  MLFLOW_TRACKING_URI: "http://mlflow-service.default.svc.cluster.local"
  ARTIFACT_ROOT: ""
  AWS_ACCESS_KEY_ID: ""
  AWS_SECRET_ACCESS_KEY: ""
```

**Note :** Notre cluster ray ici accède à notre serveur mlflow depuis l'intérieur du cluster !

2. Lancer le job:

```bash
ray job submit --runtime-env=runtime.yaml --address="http://127.0.0.1:8265" -- python train_with_ray.py
```

---

### Résultat attendu

- PostgreSQL = backend de MLflow.
- MLflow = trace et stocke les runs et modèles.
- Ray = exécute l’entraînement dans Kubernetes.
- Suivi :
  - Jobs → http://127.0.0.1:8265
  - Expériences → interface MLflow.

---

## En avant vers le Deep

### 0. Créer un dossier dans S3 pour le stockage Ray

Ray peut sauvegarder ses résultats et checkpoints dans un bucket S3.

Exemple : "s3://jedha-lead-33/ray_storage/"

---

### 1. Mise à jour du déploiement de Ray (depuis le repo ddl)

```bash
helm install kuberay-operator kuberay/kuberay-operator --version 1.4.0
```

```bash
helm install raycluster kuberay/ray-cluster --version 1.4.0 -f ray-cluster.yaml
```

Exemple ray-cluster.yaml minimal :

```yaml
head:
  enableInTreeAutoscaling: true
  resources:
    limits:
      cpu: "3"
      # To avoid out-of-memory issues, never allocate less than 2G memory for the Ray head.
      memory: "6G"
    requests:
      cpu: "3"
      memory: "6G"

worker:
  replicas: 1
  resources:
    limits:
      cpu: "2"
      memory: "5G"
    requests:
      cpu: "2"
      memory: "5G"
```

---

### 2. Rouvrir le tableau de bord Ray (tout en exposant ray hors du cluster)

```bash
kubectl port-forward --address 0.0.0.0 service/raycluster-kuberay-head-svc 8265:8265
```

---

### 3. Soumettre le job pour le premier modèle (exercice 1)

Dans un autre terminal :

1. runtime.yaml :

```yaml
---
working_dir: "./"
pip:
  - ray[train]
  - torch
  - mlflow==2.21.3
  - boto3
  - requests>=2.31.0,<3
  - torchvision
  - s3fs
env_vars:
  ARTIFACT_ROOT: ""
  AWS_ACCESS_KEY_ID: ""
  AWS_SECRET_ACCESS_KEY: ""
```

2. Lancer le job:

```bash
ray job submit --runtime-env=runtime.yaml --address="http://127.0.0.1:8265" -- python train.py
```

---

### Résultat attendu

- Ray = exécute l’entraînement dans Kubernetes.
- Mise à jour du jobs dans le dashboard ray
- Récupération des poids du modèles dans votre S3

---

### 4. Pour arrêter le job (si trop long)

Dans un nouveau terminal:

```bash
# Lister
ray job list --address="http://127.0.0.1:8265"
# Stopper
ray job stop <SUBMISSION_ID> --address="http://127.0.0.1:8265"
```

---

### 5. Passons à l'hyperparametrage (exercice 2)

Dans le terminal on reste dans ddl, il va juste falloir modifier le runtime.yaml pour y ajouter optuna :

1. runtime_hyperparameter.yaml :

```yaml
---
working_dir: "./"
pip:
  - ray[train]
  - torch
  - mlflow==2.21.3
  - boto3
  - requests>=2.31.0,<3
  - torchvision
  - s3fs
  - optuna
env_vars:
  ARTIFACT_ROOT: ""
  AWS_ACCESS_KEY_ID: ""
  AWS_SECRET_ACCESS_KEY: ""
```

2. Lancer le job:

```bash
ray job submit --runtime-env=runtime.yaml --address="http://127.0.0.1:8265" -- python train_hyperparameter.py
```

---

### Résultat attendu

- Ray = exécute l’entraînement dans Kubernetes.
- Mise à jour du jobs dans le dashboard ray
- Récupération des poids du modèles dans votre S3

---

### 6. Pour arrêter le job (si trop long)

Dans un nouveau terminal:

```bash
# Lister
ray job list --address="http://127.0.0.1:8265"
# Stopper
ray job stop <SUBMISSION_ID> --address="http://127.0.0.1:8265"
```

---

## Happy coding !
