# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model

CSV_PATH = 'data/dataset2.csv'
SEQUENCE_LENGTH = 30
TF_RANDOM_SEED = 42
np.random.seed(0)

# Load model & encoders
model = load_model('./model/model.keras', compile=False)
model.compile(optimizer='adam', loss='mse')
scaler_X = joblib.load('./model/scaler_X.pkl')
scaler_y = joblib.load('./model/scaler_y.pkl')
encoder = joblib.load('./model/encoder.pkl')

# Load dataset
df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['date'])
df.ffill(inplace=True)
problem_types = df['problem_type'].unique().tolist()

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

# Helper: dynamic workforce calculation
def calculate_dynamic_workforce(predicted_cases, problem_type, df, urgency_factor=1.0, max_workers=200):
    subset = df[df['problem_type'].str.strip().str.title() == problem_type.title()]
    if subset.empty:
        avg_cases_per_worker = 1.5
        base_staff = 1
        efficiency_factor = 0.85
        severity_score = 2
    else:
        avg_cases_per_worker = subset['reported_cases'].sum() / subset['workforce_required'].sum()
        base_staff = max(1, int(subset['workforce_required'].min() * 0.5))
        efficiency_factor = min(1.0, 0.8 + (subset['severity_score'].mean() - 2) * 0.05)
        severity_score = int(subset['severity_score'].iloc[-1])

    severity_multiplier = 1 + (severity_score ** 1.3) * 0.15
    effective_cases = predicted_cases / efficiency_factor
    workforce = (effective_cases / avg_cases_per_worker) * severity_multiplier
    workforce += base_staff
    workforce *= urgency_factor
    workforce = min(workforce, max_workers)

    return max(1, int(round(workforce)))

# Helper: predict next day cases dynamically
def predict_cases(problem_type, target_date):
    subset = df[df['problem_type'].str.strip().str.title() == problem_type.title()].sort_values('date')
    if subset.empty:
        # fallback random
        predicted_cases = np.random.randint(1, 20)
    else:
        seq_len = SEQUENCE_LENGTH
        seq = subset.tail(seq_len).copy()
        if len(seq) < seq_len:
            last_row = seq.iloc[-1].copy()
            last_date = last_row['date']
            for i in range(seq_len - len(seq)):
                new_row = last_row.copy()
                new_row['date'] = last_date + pd.Timedelta(days=i+1)
                seq = pd.concat([seq, pd.DataFrame([new_row])], ignore_index=True)

        X_seq_to_predict = seq.drop(columns=['date','reported_cases','workforce_required'], errors='ignore')
        if 'problem_type' in X_seq_to_predict.columns and 'region' in X_seq_to_predict.columns:
            cat_transformed = pd.DataFrame(
                encoder.transform(X_seq_to_predict[['problem_type','region']]),
                columns=encoder.get_feature_names_out(['problem_type','region'])
            )
            X_seq_to_predict = pd.concat([X_seq_to_predict.drop(columns=['problem_type','region']), cat_transformed], axis=1)

        X_scaled = scaler_X.transform(X_seq_to_predict)
        X_input = np.array([X_scaled])
        y_pred_scaled = model.predict(X_input, verbose=0)
        predicted_cases = max(1, int(round(scaler_y.inverse_transform(y_pred_scaled)[0,0])))

    return predicted_cases

@app.get("/problems")
def get_problems():
    return {"problems": problem_types}

@app.post("/predict")
def predict(request: PredictionRequest):
    selected_problem = request.problem
    sel_date = pd.to_datetime(request.date)

    if selected_problem not in problem_types:
        return {"error": f"Problem '{selected_problem}' not found"}

    predicted_cases = predict_cases(selected_problem, sel_date)
    predicted_workforce = calculate_dynamic_workforce(predicted_cases, selected_problem, df)

    return {
        "selected_problem": selected_problem,
        "selected_date": str(sel_date.date()),
        "predicted_cases": predicted_cases,
        "predicted_workforce": predicted_workforce
    }
