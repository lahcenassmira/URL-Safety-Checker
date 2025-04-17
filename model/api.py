from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str

@app.post("/check")
def check_url(data: URLInput):
    features = vectorizer.transform([data.url])
    prediction = model.predict(features)[0]
    label = "safe" if prediction == 0 else "suspicious"
    return {"url": data.url, "status": label}
