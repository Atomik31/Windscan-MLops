"""
Dashboard Wind Turbine - Turbine 1
"""

import io
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import boto3
import psycopg2
import mlflow
import mlflow.pyfunc
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv

st.set_page_config(
    page_title="Wind Turbine Maintenance",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🌬️ Wind Turbine Maintenance — Turbine 1")

# ============================================================================
# PATHS & S3 CONFIG
# ============================================================================

ROOT_DIR  = Path(__file__).resolve().parent.parent

# Load AWS credentials from notebook/.env (local) or env vars (HuggingFace Spaces)
_env_file = ROOT_DIR / "notebook" / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

S3_BUCKET    = "windscan"
S3_REGION    = "eu-north-1"
S3_DATA_KEY  = "data/processed/"
S3_MODEL_KEY = "models/"

MODEL_FILE        = "best_model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
DATASET_FILE      = "wind_turbine_maintenance_test_data_cleaned.csv"

# ============================================================================
# S3 LOADERS (with local fallback)
# ============================================================================

def _s3_client():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

def _pg_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        dbname=os.getenv("PGDATABASE", "neondb"),
        user=os.getenv("PGUSER", "neondb_owner"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "require")
    )

@st.cache_data(ttl=300, show_spinner="Chargement depuis Neon DB...")
def load_data(dataset_name=None):
    """Charge les données capteurs depuis Neon DB, fallback S3 puis local."""
    # Priorité 1 : Neon DB
    try:
        conn = _pg_conn()
        df = pd.read_sql("""
            SELECT id,
                   rotor_speed_rpm       AS "Rotor_Speed_RPM",
                   wind_speed_mps        AS "Wind_Speed_mps",
                   power_output_kw       AS "Power_Output_kW",
                   gearbox_oil_temp_c    AS "Gearbox_Oil_Temp_C",
                   generator_bearing_temp_c AS "Generator_Bearing_Temp_C",
                   vibration_level_mmps  AS "Vibration_Level_mmps",
                   ambient_temp_c        AS "Ambient_Temp_C",
                   humidity_pct          AS "Humidity_pct",
                   maintenance_label     AS "Maintenance_Label"
            FROM wind_turbine_sensors
            WHERE turbine_id = 1
            ORDER BY id ASC
        """, conn)
        conn.close()
        return df
    except Exception as e:
        st.warning(f"Neon DB indisponible ({e}), fallback S3...")

    # Priorité 2 : S3
    try:
        s3  = _s3_client()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY + DATASET_FILE)
        df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
        if "Turbine_ID" in df.columns:
            df = df[df["Turbine_ID"] == 1].reset_index(drop=True)
        return df
    except Exception:
        pass

    # Priorité 3 : local
    local = ROOT_DIR / "data" / "processed" / DATASET_FILE
    if local.exists():
        df = pd.read_csv(local)
        if "Turbine_ID" in df.columns:
            df = df[df["Turbine_ID"] == 1].reset_index(drop=True)
        return df

    st.error("❌ Données introuvables (Neon DB, S3 et local)")
    st.stop()

@st.cache_resource(show_spinner="Loading model from MLflow...")
def load_model(model_name=None):
    """Load pipeline (preprocessor + model) from MLflow Model Registry."""
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://atomik31-mlflow.hf.space"))
        return mlflow.pyfunc.load_model("models:/turbine_maintenance_predictor@production")
    except Exception as e:
        st.warning(f"MLflow unavailable ({e}), falling back to local model.")
        local = ROOT_DIR / "models" / "best_model.pkl"
        if local.exists():
            return joblib.load(local)
        st.error("❌ Model not found in MLflow or locally.")
        st.stop()

@st.cache_resource(show_spinner="Loading preprocessor from S3...")
def load_preprocessor(preprocessor_name):
    """Load preprocessor from S3, fallback to local."""
    try:
        s3  = _s3_client()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY + preprocessor_name)
        return joblib.load(io.BytesIO(obj["Body"].read()))
    except Exception:
        local = ROOT_DIR / "models" / preprocessor_name
        return joblib.load(local) if local.exists() else None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Model & Data")
    st.info("🤖 Model: `turbine_maintenance_predictor@production` (MLflow)")
    st.info("📂 Dataset: `wind_turbine_sensors` (Neon DB)")

    st.divider()
    st.subheader("Time Window")
    time_choice = st.radio(
        "Select time window", ["Predefined", "Custom Hours"], horizontal=True
    )

    if time_choice == "Predefined":
        time_options = {
            "All Data":      None,
            "Last 3 Hours":  3,
            "Last 6 Hours":  6,
            "Last 12 Hours": 12,
            "Last 24 Hours": 24,
            "Last 48 Hours": 48,
            "Last 72 Hours": 72,
            "Last 1 Week":   168,
            "Last 2 Weeks":  336,
        }
        selected_time = st.selectbox(
            "Choose preset", list(time_options.keys()), index=4
        )
        hours_limit = time_options[selected_time]
    else:
        hours_limit = st.slider(
            "Select hours", min_value=1, max_value=336, value=24, step=1
        )
        selected_time = f"Last {hours_limit} Hours"

# ============================================================================
# LOAD DATA & MODEL
# ============================================================================

df           = load_data(DATASET_FILE)
model        = load_model(MODEL_FILE)
preprocessor = load_preprocessor(PREPROCESSOR_FILE)

df_filtered = df.copy()
if hours_limit:
    df_filtered = df_filtered.tail(hours_limit).reset_index(drop=True)

st.write(f"📊 Time: {selected_time} | Rows: {len(df_filtered)}")

st.sidebar.subheader("Select Specific Hour")
selected_hour = st.sidebar.slider(
    "View status at hour:",
    min_value=0,
    max_value=len(df_filtered) - 1,
    value=len(df_filtered) - 1,
    step=1,
)

# ============================================================================
# FEATURE COLUMNS
# ============================================================================

exclude = {"Turbine_ID", "Timestamp", "id"}

if isinstance(model, dict) and "feature_columns" in model:
    feature_cols = model["feature_columns"]
    model_obj    = model["model"]
    scaler       = model.get("scaler", preprocessor)
elif hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
    model_obj    = model
    scaler       = preprocessor
else:
    feature_cols = [c for c in df_filtered.columns if c not in exclude]
    model_obj    = model
    scaler       = preprocessor

# ============================================================================
# SECTION 1 — REAL-TIME STATUS
# ============================================================================

st.header("🚨 Real-Time Status")

if model_obj is not None and len(df_filtered) > 0:
    try:
        X = df_filtered[feature_cols].fillna(0)
        if scaler is not None:
            X = scaler.transform(X)

        predictions = model_obj.predict(X)

        if hasattr(model_obj, "predict_proba"):
            proba      = model_obj.predict_proba(X)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.ones(len(X))

        selected_pred = int(predictions[selected_hour])
        selected_conf = float(confidence[selected_hour])

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if selected_pred == 0:
                st.success("🟢 **HEALTHY** — No action needed", icon="✅")
            elif selected_pred == 1:
                st.warning("🟡 **MAINTENANCE NEEDED** — Schedule soon", icon="⚠️")
            else:
                st.error("🔴 **CRITICAL** — Immediate action required", icon="🚨")
        col2.metric("Confidence", f"{selected_conf:.1%}")
        col3.metric("Hour #", selected_hour)

        st.divider()
        st.subheader(f"Measurements at Hour #{selected_hour}")
        row = df_filtered.iloc[selected_hour]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Wind Speed",   f"{row['Wind_Speed_mps']:.2f} m/s")
        c2.metric("Power Output", f"{row['Power_Output_kW']:.2f} kW")
        c3.metric("Rotor Speed",  f"{row['Rotor_Speed_RPM']:.0f} RPM")
        c4.metric("Vibration",    f"{row['Vibration_Level_mmps']:.2f} mm/s")
        st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        st.divider()
        st.subheader("Prediction Timeline")
        colors_map = {0: "#28a745", 1: "#ffc107", 2: "#dc3545"}
        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=list(range(len(predictions))),
            y=predictions,
            mode="markers+lines",
            marker=dict(
                size=5,
                color=[colors_map.get(int(p), "#999") for p in predictions],
                line=dict(width=1, color="white"),
            ),
            line=dict(color="rgba(0,0,0,0.1)"),
            name="Status",
        ))
        fig_tl.update_yaxes(tickvals=[0, 1, 2], ticktext=["Healthy", "Maintenance", "Critical"])
        fig_tl.update_layout(height=250, hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_tl, use_container_width=True)

        st.subheader("Statistics")
        unique, counts = np.unique(predictions, return_counts=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Healthy",     int(counts[unique == 0][0] if 0 in unique else 0))
        c2.metric("🟡 Maintenance", int(counts[unique == 1][0] if 1 in unique else 0))
        c3.metric("🔴 Critical",    int(counts[unique == 2][0] if 2 in unique else 0))

        if "Maintenance_Label" in df_filtered.columns:
            actual   = df_filtered["Maintenance_Label"].values
            accuracy = (predictions == actual).sum() / len(predictions)
            st.metric("Accuracy on Period", f"{accuracy:.1%}")

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.error("❌ Model or data not available")

# ============================================================================
# SECTION 2 — FEATURE MONITORING
# ============================================================================

st.divider()
st.header("📊 Feature Monitoring")

numeric_features = [c for c in feature_cols if c in df_filtered.columns
                    and pd.api.types.is_numeric_dtype(df_filtered[c])]

if numeric_features:
    tabs = st.tabs(numeric_features)
    for tab, feature in zip(tabs, numeric_features):
        with tab:
            fig = px.line(df_filtered, y=feature, title=feature, markers=False)
            fig.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Min",  f"{df_filtered[feature].min():.2f}")
            c2.metric("Max",  f"{df_filtered[feature].max():.2f}")
            c3.metric("Mean", f"{df_filtered[feature].mean():.2f}")
            c4.metric("Std",  f"{df_filtered[feature].std():.2f}")
else:
    st.warning("⚠️ No numeric features available for monitoring")

# ============================================================================
# SECTION 3 — RAW DATA
# ============================================================================

st.divider()
st.header("📋 Raw Data")

if st.checkbox("Show all rows"):
    st.dataframe(df_filtered, use_container_width=True, height=400)
else:
    st.dataframe(df_filtered.tail(20), use_container_width=True, height=400)

st.divider()
st.caption(f"Rows: {len(df_filtered)} | ⏰ {datetime.now().strftime('%H:%M:%S')}")
