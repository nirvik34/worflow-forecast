# app.py (fixed / defensive)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)

CSV_PATH = 'data/dataset2.csv'
SEQUENCE_LENGTH = 30
TF_RANDOM_SEED = 42
np.random.seed(0)

# Load model & encoders (keep your original paths)


@lru_cache(maxsize=1)
def load_artifacts():
    # lazy load heavy artifacts once per process
    model = load_model('./model/model.keras', compile=False)
    model.compile(optimizer='adam', loss='mse')
    scaler_X = joblib.load('./model/scaler_X.pkl')
    scaler_y = joblib.load('./model/scaler_y.pkl')
    encoder = joblib.load('./model/encoder.pkl')
    return model, scaler_X, scaler_y, encoder

# Load dataset
df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['date'])
# coerce numeric columns to numeric types (NaNs if invalid)
for c in ['reported_cases', 'workforce_required', 'severity_score']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
df.ffill(inplace=True)
# keep list of problem types as originally present (no NaN entries)
problem_types = df['problem_type'].dropna().unique().tolist()

app = FastAPI(title="Workforce Forecast API")
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    problem: str
    date: str  # YYYY-MM-DD

# Safe conversion helpers
def safe_float(x, fallback=np.nan):
    try:
        if pd.isna(x):
            return fallback
        return float(x)
    except Exception:
        return fallback

def safe_int(x, fallback=1):
    try:
        if pd.isna(x):
            return fallback
        v = int(round(float(x)))
        return v
    except Exception:
        return fallback

# Normalize problem_type matching (case-insensitive)
def find_canonical_problem(name):
    if not name or not isinstance(name, str):
        return None
    name_norm = name.strip().lower()
    for p in problem_types:
        if isinstance(p, str) and p.strip().lower() == name_norm:
            return p
    return None

# Helper: dynamic workforce calculation (defensive)
def calculate_dynamic_workforce(predicted_cases, problem_type, df, urgency_factor=1.0, max_workers=200):
    # find subset case-insensitively
    try:
        subset = df[df['problem_type'].fillna('').str.strip().str.lower() == problem_type.strip().lower()]
    except Exception:
        subset = pd.DataFrame()

    # defaults
    avg_cases_per_worker = 1.5
    base_staff = 1
    efficiency_factor = 0.85
    severity_score = 2

    if not subset.empty:
        # totals (safe)
        reported_sum = safe_float(subset['reported_cases'].sum(), fallback=np.nan)
        workforce_sum = safe_float(subset['workforce_required'].sum(), fallback=np.nan)

        # avg cases per worker -> protect division by zero and NaN
        if np.isnan(workforce_sum) or workforce_sum == 0:
            avg_cases_per_worker = 1.5
        else:
            avg_cases_per_worker = reported_sum / workforce_sum
            if not np.isfinite(avg_cases_per_worker) or avg_cases_per_worker <= 0:
                avg_cases_per_worker = 1.5

        # base staff from historical minimum workforce
        min_workforce = subset['workforce_required'].min()
        if pd.isna(min_workforce) or not np.isfinite(min_workforce):
            base_staff = 1
        else:
            base_staff = max(1, safe_int(min_workforce * 0.5, fallback=1))

        # efficiency factor from mean severity (bounded)
        severity_mean = safe_float(subset['severity_score'].mean(), fallback=2.0)
        if not np.isfinite(severity_mean):
            severity_mean = 2.0
        efficiency_factor = min(1.0, max(0.1, 0.8 + (severity_mean - 2) * 0.05))

        # severity score take last non-NaN if possible
        try:
            last_sev = subset['severity_score'].dropna().iloc[-1]
            severity_score = safe_int(last_sev, fallback=int(round(severity_mean)))
        except Exception:
            severity_score = int(round(severity_mean))

    # compute workforce (protect against NaN/div-by-zero)
    severity_multiplier = 1 + (max(0, severity_score) ** 1.3) * 0.15
    effective_cases = predicted_cases / max(0.01, efficiency_factor)
    workforce = (effective_cases / max(0.1, avg_cases_per_worker)) * severity_multiplier
    workforce += base_staff
    workforce *= urgency_factor
    # cap and ensure finite
    if not np.isfinite(workforce) or pd.isna(workforce):
        workforce = base_staff
    workforce = min(workforce, max_workers)
    return max(1, int(round(workforce)))

# Helper: predict next day cases dynamically (defensive)
def predict_cases(problem_type, target_date):
    # filter case-insensitively
    subset = df[df['problem_type'].fillna('').str.strip().str.lower() == problem_type.strip().lower()].sort_values('date')
    if subset.empty:
        # fallback random / small heuristic
        return int(np.random.randint(1, 20))

    seq_len = SEQUENCE_LENGTH
    seq = subset.tail(seq_len).copy()
    if len(seq) < seq_len:
        # pad with last known row
        last_row = seq.iloc[-1].copy()
        last_date = last_row['date']
        for i in range(seq_len - len(seq)):
            new_row = last_row.copy()
            new_row['date'] = last_date + pd.Timedelta(days=i+1)
            seq = pd.concat([seq, pd.DataFrame([new_row])], ignore_index=True)

    X_seq_to_predict = seq.drop(columns=['date', 'reported_cases', 'workforce_required'], errors='ignore')

    # handle categorical encoding safely
    cat_cols = [c for c in ['problem_type', 'region'] if c in X_seq_to_predict.columns]
    if cat_cols:
        try:
            cat_transformed = pd.DataFrame(
                encoder.transform(X_seq_to_predict[cat_cols]),
                columns=encoder.get_feature_names_out(cat_cols),
                index=X_seq_to_predict.index
            )
            X_seq_to_predict = pd.concat([X_seq_to_predict.drop(columns=cat_cols), cat_transformed], axis=1)
        except Exception as e:
            logging.warning("Encoder transform failed (%s). Dropping categorical columns as fallback.", e)
            X_seq_to_predict = X_seq_to_predict.drop(columns=cat_cols)

    try:
        X_scaled = scaler_X.transform(X_seq_to_predict)
        X_input = np.array([X_scaled])
        y_pred_scaled = model.predict(X_input, verbose=0)
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_inv = scaler_y.inverse_transform(y_pred_scaled)
        predicted_cases = max(1, safe_int(y_inv[0, 0], fallback=1))
    except Exception as e:
        logging.warning("Model/scaler prediction failed: %s. Falling back to heuristic.", e)
        last_reported = None
        if 'reported_cases' in seq.columns:
            try:
                last_reported = seq['reported_cases'].dropna().iloc[-1]
            except Exception:
                last_reported = None
        if last_reported is None or not np.isfinite(safe_float(last_reported)):
            predicted_cases = int(np.random.randint(1, 10))
        else:
            predicted_cases = max(1, safe_int(last_reported, fallback=1))

    return predicted_cases

@app.get("/problems")
def get_problems():
    # return canonical list
    return {"problems": problem_types}

@app.post("/predict")
def predict(request: PredictionRequest):
    selected_problem_raw = request.problem
    sel_date = pd.to_datetime(request.date, errors='coerce')
    if pd.isna(sel_date):
        return {"error": f"Invalid date: {request.date}"}

    canonical = find_canonical_problem(selected_problem_raw)
    if canonical is None:
        return {"error": f"Problem '{selected_problem_raw}' not found"}

    predicted_cases = predict_cases(canonical, sel_date)
    predicted_workforce = calculate_dynamic_workforce(predicted_cases, canonical, df)

    return {
        "selected_problem": canonical,
        "selected_date": str(sel_date.date()),
        "predicted_cases": int(predicted_cases),
        "predicted_workforce": int(predicted_workforce)
    }
