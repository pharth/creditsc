from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import sklearn
import uvicorn

app = FastAPI()


with open('model_linear.pkl', 'rb') as file:
    model_linear = joblib.load(file)

with open('lgb_classifier.pkl', 'rb') as file:
    lgb_classifier = joblib.load(file)


loan_status_db = {}


class ModelInput(BaseModel):
    feature2: float
    feature3: float
    loan_id: str  


class LoanStatus(BaseModel):
    loan_id: str
    due_date: datetime
    status: str  

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}

@app.post('/predict')
def predict(loan_status: LoanStatus):
    data = loan_status.dict()
    loan_id = data['loan_id']
    status = data['status']
    due_date = data['due_date']

    
    days_since_due = (datetime.now() - due_date).days

    
    
    if status == 'unpaid':
        if days_since_due <= 30:
            feature1 = 1  
        else:
            feature1 = 2  
    else:
        feature1 = 0  

    prediction = model_linear.predict([[feature1, loan_status.feature2, loan_status.feature3]])
    loan_status_db[loan_id] = status

    return {
        'loan_id': loan_id,
        'status': status,
        'prediction': prediction[0]
    }

@app.post('/pay_loan')
def pay_loan(loan_status: LoanStatus):
    loan_id = loan_status.loan_id
    
    if loan_id in loan_status_db:
        loan_status_db[loan_id] = 'paid'
        return {'loan_id': loan_id, 'status': 'paid'}
    else:
        raise HTTPException(status_code=404, detail='Loan ID not found')

@app.post('/label')
def label(input_data: ModelInput):
    data = input_data.dict()
    loan_id = data['loan_id']
    features = [[loan_status_db.get(loan_id, 0), data['feature2'], data['feature3']]]  
    label = lgb_classifier.predict(features)
    loan_label_db[loan_id] = int(label[0])  
    return {
        'loan_id': loan_id,
        'label': int(label[0])
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
