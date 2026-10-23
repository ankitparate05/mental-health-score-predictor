# Student Aura Predictor

A machine learning web app that predicts a student's mental health score from their daily habits
(sleep, study, physical activity, screen time, stress and more).

- **Model:** Random Forest regressor in a scikit-learn pipeline (one-hot encoding + forest), trained on 5,000 student survey responses
- **API:** FastAPI, one endpoint: `POST /predict`
- **Frontend:** a single `index.html` page (GitHub Pages)

**Live demo:** https://YOUR-USERNAME.github.io/YOUR-REPO/
**API docs:** https://YOUR-APP-NAME.onrender.com/docs

> The API runs on a free server that sleeps when idle, so the first request can take up to a minute.

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
# mental-health-score-predictor
