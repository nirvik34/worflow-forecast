
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
import logging
import requests
from functools import lru_cache

logging.basicConfig(level=logging.INFO)

# ======================
# CONFIG
# ======================
CSV_PATH = "data/dataset2.csv"
SEQUENCE_LENGTH = 30
TF_RANDOM_SEED = 42
np.random.seed(TF_RANDOM_SEED)

# ======================
# LOAD ARTIFACTS LAZILY
# ======================
@lru_cache(maxsize=1)
def load_artifacts():
    try:
        model = load_model("./model/model.keras", compile=False)
        model.compile(optimizer="adam", loss="mse")
        scaler_X = joblib.load("./model/scaler_X.pkl")
        scaler_y = joblib.load("./model/scaler_y.pkl")
        encoder = joblib.load("./model/encoder.pkl")
        return model, scaler_X, scaler_y, encoder
    except Exception as e:
        logging.error("Artifact load failed: %s", e)
        return None, None, None, None

model, scaler_X, scaler_y, encoder = load_artifacts()

# ======================
# LOAD DATASET
# ======================
try:
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["reported_cases", "workforce_required", "severity_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.ffill(inplace=True)
    problem_types = df["problem_type"].dropna().unique().tolist()
except Exception as e:
    logging.error("Dataset load failed: %s", e)
    df = pd.DataFrame()
    problem_types = []

# ======================
# FASTAPI SETUP
# ======================
app = FastAPI(title="Workforce Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    problem: str
    date: str  # YYYY-MM-DD
    lat: float = 28.6139
    lon: float = 77.2090

# ======================
# HELPERS
# ======================
def safe_float(x, fallback=np.nan):
    try:
        if pd.isna(x):
            return fallback
        return float(x)
    except Exception:
        return fallback

def safe_int(x, fallback=1):
    try:
        return int(round(float(x)))
    except Exception:
        return fallback

def find_canonical_problem(name: str):
    if not name or not isinstance(name, str):
        return None
    name_norm = name.strip().lower()
    for p in problem_types:
        if isinstance(p, str) and p.strip().lower() == name_norm:
            return p
    return None

def get_weather_forecast(lat, lon, target_date):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
        f"&start_date={target_date}&end_date={target_date}&timezone=UTC"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        rainfall = float(daily.get("precipitation_sum", [0])[0])
        score = 1.0
        if rainfall > 20:
            score = 1.5
        elif rainfall > 5:
            score = 1.2
        return {"rainfall": rainfall, "score": score}
    except Exception as e:
        logging.warning("Weather fetch failed: %s", e)
        return {"rainfall": 0.0, "score": 1.0}

def calculate_dynamic_workforce(predicted_cases, problem_type, df, weather=None,
                                urgency_factor=1.0, max_workers=200):
    subset = df[df["problem_type"].fillna("").str.strip().str.lower() == problem_type.strip().lower()]
    avg_cases_per_worker, base_staff, efficiency_factor, severity_score = 1.5, 1, 0.85, 2

    if not subset.empty:
        reported_sum = safe_float(subset["reported_cases"].sum())
        workforce_sum = safe_float(subset["workforce_required"].sum())
        avg_cases_per_worker = reported_sum / workforce_sum if workforce_sum and workforce_sum > 0 else 1.5
        base_staff = max(1, safe_int(subset["workforce_required"].min() * 0.5, 1))
        severity_mean = safe_float(subset["severity_score"].mean(), 2.0)
        efficiency_factor = min(1.0, max(0.1, 0.8 + (severity_mean - 2) * 0.05))
        try:
            severity_score = safe_int(subset["severity_score"].dropna().iloc[-1], int(round(severity_mean)))
        except Exception:
            severity_score = int(round(severity_mean))

    if weather:
        severity_score = int(round(severity_score * weather.get("score", 1.0)))
        if weather.get("rainfall", 0) > 5:
            efficiency_factor *= 0.95

    severity_multiplier = 1 + (max(0, severity_score) ** 1.3) * 0.15
    effective_cases = predicted_cases / max(0.01, efficiency_factor)
    workforce = (effective_cases / max(0.1, avg_cases_per_worker)) * severity_multiplier
    workforce = min(workforce * urgency_factor + base_staff, max_workers)
    return max(1, safe_int(workforce))
# --- updated predict_cases ---
def predict_cases(problem_type, target_date):
    subset = df[df['problem_type'].fillna('').str.strip().str.lower() == problem_type.strip().lower()].sort_values('date')
    if subset.empty:
        return int(np.random.randint(1, 20))

    seq_len = SEQUENCE_LENGTH
    seq = subset.tail(seq_len).copy()

    # if target_date is after last known date, extend step by step until target_date
    last_known_date = seq['date'].iloc[-1]
    while last_known_date < target_date:
        last_row = seq.iloc[-1].copy()
        new_date = last_row["date"] + pd.Timedelta(days=1)
        new_row = last_row.copy()
        new_row["date"] = new_date
        # update date-based features
        if "day_of_year" in seq.columns:
            new_row["day_of_year"] = new_date.timetuple().tm_yday
        if "weekday" in seq.columns:
            new_row["weekday"] = new_date.weekday()
        if "month" in seq.columns:
            new_row["month"] = new_date.month
        if "dayofyear_sin" in seq.columns and "dayofyear_cos" in seq.columns:
            doy = new_date.timetuple().tm_yday
            new_row["dayofyear_sin"] = np.sin(2 * np.pi * doy / 365.25)
            new_row["dayofyear_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # --- predict next step using seq ---
        X_seq_to_predict = seq.tail(seq_len).drop(columns=['date','reported_cases','workforce_required'], errors='ignore')

        # categorical encoding safe
        cat_cols = [c for c in ['problem_type', 'region'] if c in X_seq_to_predict.columns]
        if cat_cols:
            try:
                cat_transformed = pd.DataFrame(
                    encoder.transform(X_seq_to_predict[cat_cols]),
                    columns=encoder.get_feature_names_out(cat_cols),
                    index=X_seq_to_predict.index
                )
                X_seq_to_predict = pd.concat([X_seq_to_predict.drop(columns=cat_cols), cat_transformed], axis=1)
            except Exception:
                X_seq_to_predict = X_seq_to_predict.drop(columns=cat_cols)

        try:
            X_scaled = scaler_X.transform(X_seq_to_predict)
            X_input = np.array([X_scaled])
            y_pred_scaled = model.predict(X_input, verbose=0)
            y_inv = scaler_y.inverse_transform(y_pred_scaled)
            predicted_cases = max(1, safe_int(y_inv[0, 0], fallback=1))
        except Exception:
            # fallback = repeat last known
            predicted_cases = safe_int(seq['reported_cases'].iloc[-1], fallback=1)

        new_row["reported_cases"] = predicted_cases
        seq = pd.concat([seq, pd.DataFrame([new_row])], ignore_index=True)
        last_known_date = new_date

    # when target_date is reached, return last predicted
    try:
        final_pred = int(seq.loc[seq['date'] == target_date, 'reported_cases'].iloc[-1])
    except Exception:
        final_pred = safe_int(seq['reported_cases'].iloc[-1], fallback=1)

    return final_pred



# ======================
# ROUTES
# ======================
@app.get("/problems")
def get_problems():
    return {"problems": problem_types}

@app.post("/predict")
def predict(req: PredictionRequest):
    sel_date = pd.to_datetime(req.date, errors="coerce")
    if pd.isna(sel_date):
        return {"error": f"Invalid date: {req.date}"}
    if sel_date.date() < datetime.today().date():
        return {"error": "Past dates not allowed"}

    canonical = find_canonical_problem(req.problem)
    if not canonical:
        return {"error": f"Problem '{req.problem}' not found"}

    predicted_cases = predict_cases(canonical, sel_date)
    weather = get_weather_forecast(req.lat, req.lon, sel_date.date())
    predicted_workforce = calculate_dynamic_workforce(predicted_cases, canonical, df, weather)

    return {
        "problem": canonical,
        "date": str(sel_date.date()),
        "predicted_cases": predicted_cases,
        "predicted_workforce": predicted_workforce,
        "weather": weather,
    }
