from src import utils, preprocessing
import pandas as pd
from fastapi import FastAPI, Response, status
from pydantic import BaseModel

import pickle
import joblib


# Buat instance dari FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Your API is UP. Good job"}

# Buat path menuju best model
model_path = 'models/trained_RandomForest.pkl'

# Muat model
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


# Buat path menuju Ohe Object
ohe_files = {
    'ohe_home_ownership' : 'models/ohe_home_ownership.pkl',
    'ohe_loan_intent'    : 'models/ohe_loan_intent.pkl',
    'ohe_loan_grade'     : 'models/ohe_loan_grade.pkl',
    'ohe_default_on_file': 'models/ohe_default_on_file.pkl'
}

ohe_objects = {}

# Muat Ohe Object
for ohe_name, ohe_path in ohe_files.items():
    try:
        ohe_objects[ohe_name] = joblib.load(ohe_path)
        print(f"{ohe_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading {ohe_name} from {ohe_path}: {e}")

# Verifikasi proses muat
for key, value in ohe_objects.items():
    print(f"{key}: {value}")

# Buat class Item yang mewarisi BaseModel
class Item(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

@app.post("/predict")
async def predict(item: Item, response:Response):
    try:
        # Konversi input ke dalam DataFrame
        input_data = pd.DataFrame([item.dict()])

        # Lakukan preprocessing
        input_data = preprocessing.ohe_transform(input_data, 'person_home_ownership', 'home_ownership', ohe_objects['ohe_home_ownership'])
        input_data = preprocessing.ohe_transform(input_data, 'loan_intent', 'loan_intent', ohe_objects['ohe_loan_intent'])
        input_data = preprocessing.ohe_transform(input_data, 'loan_grade', 'loan_grade', ohe_objects['ohe_loan_grade'])
        input_data = preprocessing.ohe_transform(input_data, 'cb_person_default_on_file', 'default_on_file', ohe_objects['ohe_default_on_file'])
        
        # Prediksi probabilitas
        proba = model.predict_proba(input_data)[:, 1]

        # Tentukan kelas berdasarkan threshold
        threshold  = 0.37  # Sesuaikan threshold
        prediction = (proba >= threshold).astype(int)

        # Pendefinisian status: apa artinya 0 dan 1
        if int(prediction[0]) == 1:
            status = "Probably Default"
        else:
            status = "Probably Non Default"

        return {
        "Prediction" : status,
        "Probability": float(proba[0])
        }
        
    except Exception as e:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'status': 'error', 'message': 'model is not found', 'detail_error': str(e)}

