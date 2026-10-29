# Student Aura Predictor

A machine learning web app that predicts a student's mental health score from their daily habits
(sleep, study, physical activity, screen time, stress and more).

- **Model:** Random Forest regressor in a scikit-learn pipeline (one-hot encoding + forest), trained on 5,000 student survey responses
- **API:** FastAPI, one endpoint: `POST /predict`
- **Frontend:** a single `index.html` page

## Example

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "Age": 21, "Gender": "Female", "Country": "India", "Academic_Level": "Graduate",
  "Most_Used_Platform": "LinkedIn", "Purpose_Of_Use": "Education",
  "Avg_Daily_Usage_Hours": 1.5, "Daily_Unlocks": 30, "Study_Hours": 5,
  "Physical_Activity_Hours": 1.5, "Sleep_Hours_Per_Night": 8, "Stress_Level": "Low"
}'
# {"predicted_mental_health_score": 8.03}
```

Inputs are validated by Pydantic (e.g. hours between 0 and 24, allowed categories only); invalid requests get a clear 422 error.

## Run it locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Then open `index.html` in your browser, or visit http://127.0.0.1:8000/docs.

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app and input validation |
| `Mental_Health_Model.pkl` | Trained scikit-learn pipeline |
| `index.html` | Frontend |
| `ML_Project (1).ipynb` | Data cleaning, EDA, model training |
| `requirements.txt`, `render.yaml` | Deployment config |

## Notes

This is a portfolio project. Predictions are estimates from survey data, not a diagnosis.
