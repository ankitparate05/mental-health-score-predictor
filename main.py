from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

model = joblib.load('Mental_Health_Model.pkl')

app = FastAPI(title='Mental Health Score Predictor')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Same 10 groups used during training (everything else becomes 'Other')
TOP_COUNTRIES = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK',
                 'Germany', 'Mexico', 'Turkey', 'France']


# What the user sends in
class StudentData(BaseModel):
    Age: int = Field(..., ge=10, le=100)
    Gender: Literal['Male', 'Female']
    Country: str
    Academic_Level: Literal['Undergraduate', 'Graduate', 'High School']
    Most_Used_Platform: Literal[
        'Instagram', 'TikTok', 'Facebook', 'LinkedIn', 'YouTube', 'Twitter',
        'Snapchat', 'WhatsApp', 'LINE', 'VKontakte', 'KakaoTalk', 'WeChat'
    ]
    Purpose_Of_Use: Literal['Networking', 'Education', 'Entertainment', 'News']
    Avg_Daily_Usage_Hours: float = Field(..., ge=0, le=24)
    Daily_Unlocks: int = Field(..., ge=0)
    Study_Hours: float = Field(..., ge=0, le=24)
    Physical_Activity_Hours: float = Field(..., ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(..., ge=0, le=24)
    Stress_Level: Literal['Low', 'Medium', 'High', 'Very High']


# What we send back
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


@app.get('/')
def greet():
    return {'message': 'welcome Ankit'}


@app.post('/predict', response_model=PredictionResponse)
def predict(data: StudentData):
    grouped_country = data.Country if data.Country in TOP_COUNTRIES else 'Other'

    # Only the 12 columns the model was trained on
    input_row = pd.DataFrame([{
        'Study_Hours': data.Study_Hours,
        'Age': data.Age,
        'Avg_Daily_Usage_Hours': data.Avg_Daily_Usage_Hours,
        'Daily_Unlocks': data.Daily_Unlocks,
        'Physical_Activity_Hours': data.Physical_Activity_Hours,
        'Sleep_Hours_Per_Night': data.Sleep_Hours_Per_Night,
        'Stress_Level': data.Stress_Level,
        'Gender': data.Gender,
        'Academic_Level': data.Academic_Level,
        'Most_Used_Platform': data.Most_Used_Platform,
        'Purpose_Of_Use': data.Purpose_Of_Use,
        'Grouped_country': grouped_country,
    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(
        predicted_mental_health_score=round(float(prediction), 2)
    )