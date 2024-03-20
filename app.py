import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pickle
import tensorflow
import sklearn


app = FastAPI()
with open('model_linear.pkl', 'rb') as file:
    model = pickle.load(file)

# This will act as our in-memory 'database' for loan statuses
loan_status_db = {}

class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    loan_id: str  # Added a loan identifier

class LoanStatus(BaseModel):
    loan_id: str
    status: str  # Status can be 'unpaid' or 'paid'

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}

@app.post('/predict')
def predict(input_data: ModelInput):
    data = input_data.dict()
    prediction = model.predict([[data['feature1'], data['feature2'], data['feature3']]])
    # After prediction, update the loan status as 'unpaid' initially
    loan_status_db[data['loan_id']] = 'unpaid'
    return {
        'prediction': prediction[0],
        'loan_id': data['loan_id'],
        'initial_status': 'unpaid'
    }

@app.post('/pay_loan')
def pay_loan(loan_status: LoanStatus):
    loan_status_data = loan_status.dict()
    loan_id = loan_status_data['loan_id']
    # Check if the loan_id exists in our 'database'
    if loan_id in loan_status_db:
        # Update the status of the loan to 'paid'
        loan_status_db[loan_id] = 'repaid'
        return {'loan_id': loan_id, 'status': 'repaid'}
    else:
        return {'error': 'Loan ID not found'}

if __name__ == '_main_':
    uvicorn.run(app, host='127.0.0.1', port=8000)