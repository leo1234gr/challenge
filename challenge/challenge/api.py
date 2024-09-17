import fastapi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from challenge.model import DelayModel

app = FastAPI()

# Cargar el modelo
model = DelayModel()
model.load_model('model.json')

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class PredictionRequest(BaseModel):
    flights: list[FlightData]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> dict:
    try:
        data = pd.DataFrame([flight.dict() for flight in request.flights])
        features = model.preprocess(data)
        predictions = model.predict(features)
        return {"predict": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
