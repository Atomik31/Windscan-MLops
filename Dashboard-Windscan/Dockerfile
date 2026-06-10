# ============================================================
# Dockerfile — Wind Turbine Dashboard (Streamlit)
# ============================================================

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 📌 COPIER le config.toml AVANT de passer en user non-root
COPY .streamlit/config.toml /app/.streamlit/config.toml
COPY app.py .

# Créer le répertoire .streamlit avec les bonnes permissions
RUN mkdir -p /app/.streamlit && chown -R user:user /app/.streamlit

USER user

EXPOSE 7860

CMD ["streamlit", "run", "app.py"]