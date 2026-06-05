# API de prédiction de maintenance des éoliennes

API REST exposant le modèle de maintenance prédictive des éoliennes dans le cadre du pipeline MLOps. Déployée sur Hugging Face Spaces via Docker, elle est appelée ligne par ligne par le DAG Airflow `etl_WINDSCAN_dag` pour générer des prédictions sur les données capteurs des turbines.

Le modèle est un pipeline scikit-learn (preprocessing + RandomForest) entraîné avec Ray sur Kubernetes, versionné et enregistré dans MLflow sous le nom `turbine_maintenance_predictor`. Il est chargé dynamiquement au démarrage depuis le registre MLflow.

## Endpoints

### `GET /health`

Retourne le statut de l'API et confirme si le modèle a bien été chargé depuis MLflow.

**Réponse :**
```json
{
  "status": "ok",
  "model_uri": "models:/turbine_maintenance_predictor@staging",
  "model_loaded": true
}
```

---

### `POST /predict`

Prédit si une éolienne nécessite une intervention de maintenance (`1`) ou non (`0`).

**Corps de la requête** (JSON) :

| Champ | Type | Description |
|---|---|---|
| `Hour_Index` | int/float | Index temporel de la mesure |
| `Turbine_ID` | int | Identifiant de la turbine |
| `Rotor_Speed_RPM` | float | Vitesse de rotation du rotor (tr/min) |
| `Wind_Speed_mps` | float | Vitesse du vent (m/s) |
| `Power_Output_kW` | float | Puissance produite (kW) |
| `Gearbox_Oil_Temp_C` | float | Température de l'huile de boîte de vitesses (°C) |
| `Generator_Bearing_Temp_C` | float | Température du palier du générateur (°C) |
| `Vibration_Level_mmps` | float | Niveau de vibration (mm/s) |
| `Ambient_Temp_C` | float | Température ambiante (°C) |
| `Humidity_pct` | float | Taux d'humidité (%) |
| `Maintenance_Label` | int | Label de maintenance historique |

**Exemple de requête :**
```bash
curl -X POST https://atomik31-model-serv-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Hour_Index": 17513,
    "Turbine_ID": 1,
    "Rotor_Speed_RPM": 15.13,
    "Wind_Speed_mps": 8.41,
    "Power_Output_kW": 1424.76,
    "Gearbox_Oil_Temp_C": 58.11,
    "Generator_Bearing_Temp_C": 79.26,
    "Vibration_Level_mmps": 2.18,
    "Ambient_Temp_C": 8.29,
    "Humidity_pct": 60.39,
    "Maintenance_Label": 0
  }'
```

**Réponse :**
```json
{
  "prediction": 1
}
```

`1` = maintenance requise, `0` = aucune maintenance nécessaire.

## Rôle dans le pipeline MLOps

Cette API est le point d'inférence du modèle de maintenance des éoliennes :

1. **Entraînement** — Modèle entraîné avec Ray (distribué sur Kubernetes) et suivi dans MLflow sous `turbine_maintenance_predictor`
2. **Exposition** — Cette API charge le modèle depuis MLflow au démarrage et expose `/predict`
3. **Inférence batch** — Le DAG Airflow `etl_WINDSCAN_dag` appelle cette API pour chaque nouvelle ligne de données capteur, sauvegarde les prédictions sur S3, puis les charge dans PostgreSQL (Neon DB)

## Variables d'environnement (Hugging Face Secrets)

| Variable | Description |
|---|---|
| `MLFLOW_TRACKING_URI` | URL du serveur MLflow |
| `MLFLOW_REGISTERED_MODEL_NAME` | Nom du modèle enregistré (`turbine_maintenance_predictor`) |
| `MLFLOW_MODEL_ALIAS` | Alias du modèle à charger (`staging`, `production`) |
