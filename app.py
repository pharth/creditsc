import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pickle
import tensorflow as tf
import sklearn
import lightgbm

app = FastAPI()


with open('model_linear.pkl', 'rb') as file:
    model_linear = pickle.load(file)


with open('lgb_classifier.pkl', 'rb') as file:
    lgb_classifier = pickle.load(file)


loan_status_db = {}
loan_label_db = {}


class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    loan_id: str  


class LoanStatus(BaseModel):
    loan_id: str
    status: str  


class LoanLabel(BaseModel):
    loan_id: str
    label: str  

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}


#The loan object contains a due date in date format and their will be a status if its repaid or not ,
#if the due date has very recently passed away (and its not repaid), enter 1 in feature 1 
#and if it has passed a long ago enter 2 in the feature 

@app.post('/predict')
def predict(input_data: ModelInput):
    data = input_data.dict()
    prediction = model_linear.predict([[data['feature1'], data['feature2'], data['feature3']]])
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
    if loan_id in loan_status_db:
        loan_status_db[loan_id] = 'repaid'  
        return {'loan_id': loan_id, 'status': 'repaid'}
    else:
        return {'error': 'Loan ID not found'}
    

@app.post('/label')
def label(input_data: ModelInput):
    data = input_data.dict()
    features = [[data['feature1'], data['feature2'], data['feature3']]]  
    label = lgb_classifier.predict(features)
    loan_label_db[data['loan_id']] = int(label[0])  
    return {
        'loan_id': data['loan_id'],
        'label': int(label[0])  
    }



if __name__ == '_main_':  
    uvicorn.run(app, host='127.0.0.1', port=8000)
