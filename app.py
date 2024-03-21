from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
from helpers.db import get_db

app = FastAPI()

with open('model_linear.pkl', 'rb') as file:
    model_linear = joblib.load(file)

with open('lgb_classifier.pkl', 'rb') as file:
    lgb_classifier = joblib.load(file)

class ModelInput(BaseModel):
    feature2: float
    feature3: float
    loan_id: str  

class LoanStatus(BaseModel):
    loan_id: str
    due_date: int
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
    
    due_date = datetime.fromtimestamp(due_date)

    days_since_due = (datetime.now() - due_date).days
    
    if status == 'unpaid':
        if days_since_due <= 30:
            feature1 = 1  
        else:
            feature1 = 2  
    else:
        feature1 = 0 
        
    with get_db() as cursor:
        row = cursor.execute("SELECT * FROM LoanUser WHERE loanID = ?", (loan_id,)).fetchone()
        if row is None:
            feature2 = 0
            feature3 = 0
            cursor.execute("INSERT INTO LoanUser (loanID, status, dueDate, feature1, feature2, feature3) VALUES (?, ?, ?, ?, ?, ?)", (loan_id, status, due_date, feature1, feature2, feature3))
        else:
            feature2 = row[8]
            feature3 = row[9]
            cursor.execute("UPDATE LoanUser SET status = ? WHERE loanID = ?", (status, loan_id))
            
    prediction = model_linear.predict([[feature1, feature2, feature3]])
        
    return {
        'loan_id': loan_id,
        'status': status,
        'prediction': prediction[0]
    }

@app.post('/pay_loan')
def pay_loan(loan_status: LoanStatus):
    loan_id = loan_status.loan_id
    
    with get_db() as cursor:
        row = cursor.execute("SELECT * FROM LoanUser WHERE loanID = ?", (loan_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail='Loan ID not found')
        cursor.execute("UPDATE LoanUser SET status = ? WHERE loanID = ?", ("paid", loan_id))
    return {'loan_id': loan_id, 'status': 'paid'}

@app.post('/label')
def label(input_data: ModelInput):
    data = input_data.dict()
    loan_id = data['loan_id']
    with get_db() as cursor:
        row = cursor.execute("SELECT feature1 FROM LoanUser WHERE loanID = ?", (loan_id,)).fetchone()
        if row is not None:
            feature1 = row[0]
        else:
            feature1 = 0
            
    features = [[feature1, data['feature2'], data['feature3']]]  
    label = lgb_classifier.predict(features)

    with get_db() as cursor:
        cursor.execute("UPDATE LoanUser SET feature1 = ? WHERE loanID = ?", (int(label[0]), loan_id))

    return {
        'loan_id': loan_id,
        'label': int(label[0])
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
