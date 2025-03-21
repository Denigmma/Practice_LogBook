from inference import load_model, predict, preprocess_input
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os

# Путь к сохранённой модели (предполагается, что модель сохранена в формате HDF5)
MODEL_PATH = "../model/saved_models/gru_model.h5"
model = load_model(MODEL_PATH)

app = FastAPI()

# Глобальный словарь для хранения последних предсказаний для каждой установки
# Для каждой установки сохраняется структура: {"probability": float, "timestamp": str}
latest_predictions = {}

class PredictionRequest(BaseModel):
    """
    Модель входного запроса.
    Ожидается, что поле data — это список длиной 20, где каждая запись содержит 5 числовых значений:
    [segment_code, pressure, temperature, vibration, flow_rate]
    """
    data: list

@app.post("/predict")
def predict_failure(request: PredictionRequest):
    # Проверяем, что последовательность имеет длину 20
    if len(request.data) != 20:
        raise HTTPException(status_code=400, detail="Длина последовательности должна быть 20")
    # Проверяем, что каждая запись содержит 5 признаков
    for idx, row in enumerate(request.data):
        if len(row) != 5:
            raise HTTPException(status_code=400, detail=f"Запись {idx} должна содержать 5 признаков")
    try:
        failure_probability = predict(model, request.data)
        return {"failure_probability": failure_probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Новый эндпоинт для обновления последних предсказаний
class UpdateRequest(BaseModel):
    sensor: str
    probability: float
    timestamp: str

@app.post("/update")
def update_prediction(update: UpdateRequest):
    latest_predictions[update.sensor] = {"probability": update.probability, "timestamp": update.timestamp}
    return {"status": "ok"}

# Эндпоинт для отдачи последних предсказаний в виде JSON
@app.get("/predictions")
def get_predictions():
    return latest_predictions

# Эндпоинт, отдающий HTML-страницу дашборда
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    dashboard_file = "dashboard.html"
    if not os.path.exists(dashboard_file):
        raise HTTPException(status_code=404, detail="Dashboard file not found")
    with open(dashboard_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
