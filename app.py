from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
import joblib

# === FastAPI instance ===
app = FastAPI(title="Workforce Prediction API", version="1.0")

# === Load model & scalers at startup ===
try:
    model = load_model('./model/model.keras', compile=False)
    model.compile(optimizer='adam', loss='mse')
    scaler_X = joblib.load('./model/scaler_X.pkl')
    scaler_y = joblib.load('./model/scaler_y.pkl')
    encoder = joblib.load('./model/encoder.pkl')
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to load model or scaler: {e}")

# === Input schema ===
class PredictionRequest(BaseModel):
    problem_type: str
    target_date: str  # YYYY-MM-DD

# === Helper functions ===
def validate_date(target_date_str):
    try:
        target_date = pd.Timestamp(target_date_str)
    except:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    if target_date.date() < datetime.today().date():
        raise HTTPException(status_code=400, detail="Past dates are not allowed. Select a future date.")
    return target_date

def safe_numeric(value, default=0.0):
    if pd.isna(value) or np.isinf(value):
        return default
    return value

def calculate_dynamic_workforce(predicted_cases, severity_score=2, avg_cases_per_worker=1.5, urgency_factor=1.0):
    workforce = (predicted_cases / avg_cases_per_worker)
    workforce *= (1 + 0.2 * (severity_score - 1))
    workforce *= urgency_factor
    return max(1, int(round(workforce)))

def predict_random(problem_type):
    base_cases_dict = {
        "Garbage & Waste": (10, 50),
        "Water Supply": (5, 20),
        "Road Maintenance": (1, 10),
        "Electricity": (2, 15),
        "Public Safety": (1, 8)
    }
    min_cases, max_cases = base_cases_dict.get(problem_type, (1, 20))
    predicted_cases = np.random.randint(min_cases, max_cases + 1)
    severity_score = np.random.randint(1, 5)
    predicted_workforce = calculate_dynamic_workforce(predicted_cases, severity_score)
    return predicted_cases, predicted_workforce

# === Core prediction logic ===
def predict(problem_type_input, target_date_str):
    target_date = validate_date(target_date_str)
    try:
        # Load dataset
        df = pd.read_csv("data/dataset2.csv")
        df['date'] = pd.to_datetime(df['date'])
        df.fillna(method='ffill', inplace=True)

        # Filter by problem type
        subset = df[df['problem_type'].str.strip().str.title() == problem_type_input.title()].sort_values('date')
        if subset.empty:
            raise ValueError(f"No historical data for {problem_type_input}")

        # Prepare last sequence for LSTM
        seq_len = 30
        seq = subset.tail(seq_len).copy()
        if len(seq) < seq_len:
            last_row = seq.iloc[-1].copy()
            last_date = last_row['date']
            for i in range(seq_len - len(seq)):
                new_row = last_row.copy()
                new_row['date'] = last_date + pd.Timedelta(days=i+1)
                seq = pd.concat([seq, pd.DataFrame([new_row])], ignore_index=True)

        # Prepare features for prediction
        X_seq_to_predict = seq.drop(columns=['date','reported_cases','workforce_required'], errors='ignore')
        # Encode categorical
        if 'problem_type' in X_seq_to_predict.columns and 'region' in X_seq_to_predict.columns:
            cat_transformed = pd.DataFrame(encoder.transform(X_seq_to_predict[['problem_type','region']]),
                                           columns=encoder.get_feature_names_out(['problem_type','region']))
            X_seq_to_predict = pd.concat([X_seq_to_predict.drop(columns=['problem_type','region']), cat_transformed], axis=1)

        # Scale and reshape
        X_scaled = scaler_X.transform(X_seq_to_predict)
        X_input = np.array([X_scaled])
        y_pred_scaled = model.predict(X_input, verbose=0)
        predicted_cases = safe_numeric(int(round(scaler_y.inverse_transform(y_pred_scaled)[0,0])), 1)

        # Severity score from last known row
        severity_score = int(seq['severity_score'].iloc[-1]) if 'severity_score' in seq.columns else 2
        predicted_workforce = calculate_dynamic_workforce(predicted_cases, severity_score)

        return predicted_cases, predicted_workforce

    except Exception as e:
        # Fallback to random prediction
        return predict_random(problem_type_input)

# === API endpoint ===
@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    cases, workforce = predict(request.problem_type, request.target_date)
    return {
        "date": request.target_date,
        "problem_type": request.problem_type,
        "predicted_reported_cases": cases,
        "predicted_workforce_required": workforce
    }
