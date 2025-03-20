# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from inference import load_model, predict, preprocess_input
# import uvicorn
#
# app = FastAPI()
#
# # Путь к сохранённой модели (предполагается, что модель сохранена в формате HDF5)
# MODEL_PATH = "../model/saved_models/gru_model.h5"
# model = load_model(MODEL_PATH)
#
# class PredictionRequest(BaseModel):
#     """
#     Модель входного запроса.
#     Ожидается, что поле data — это список длиной 20, где каждая запись содержит 5 числовых значений:
#     [segment_code, pressure, temperature, vibration, flow_rate]
#     """
#     data: list
#
# @app.post("/predict")
# def predict_failure(request: PredictionRequest):
#     # Проверяем, что последовательность имеет длину 20
#     if len(request.data) != 20:
#         raise HTTPException(status_code=400, detail="Длина последовательности должна быть 20")
#     # Проверяем, что каждая запись содержит 5 признаков
#     for idx, row in enumerate(request.data):
#         if len(row) != 5:
#             raise HTTPException(status_code=400, detail=f"Запись {idx} должна содержать 5 признаков")
#     try:
#         failure_probability = predict(model, request.data)
#         return {"failure_probability": failure_probability}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


from inference import load_model, predict, preprocess_input
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


# Путь к сохранённой модели (предполагается, что модель сохранена в формате HDF5)
MODEL_PATH = "../model/saved_models/gru_model.h5"
model = load_model(MODEL_PATH)

app = FastAPI()

# Глобальный словарь для хранения последних предсказаний для каждой установки
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

@app.post("/update")
def update_prediction(update: UpdateRequest):
    latest_predictions[update.sensor] = update.probability
    return {"status": "ok"}

# Эндпоинт для отдачи последних предсказаний в виде JSON
@app.get("/predictions")
def get_predictions():
    return latest_predictions

# Эндпоинт, отдающий HTML-страницу дашборда
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Дашборд установок</title>
      <style>
        body { font-family: Arial, sans-serif; }
        .dashboard {
          display: flex;
          justify-content: space-around;
          align-items: flex-end;
          height: 400px;
          margin: 20px;
        }
        .column {
          width: 8%;
          text-align: center;
          padding: 5px;
          border: 1px solid #ccc;
          border-radius: 4px;
          transition: background-color 0.5s;
        }
        .column.green { background-color: #90ee90; }
        .column.red { background-color: #ffcccb; }
      </style>
    </head>
    <body>
      <h1>Мониторинг установок</h1>
      <div class="dashboard" id="dashboard">
        <!-- Колонки появятся здесь -->
      </div>
      <script>
        const sensors = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"];
        function updateDashboard(predictions) {
          const dashboard = document.getElementById("dashboard");
          dashboard.innerHTML = "";
          sensors.forEach(sensor => {
            const prob = predictions[sensor] !== undefined ? predictions[sensor] : 0;
            const col = document.createElement("div");
            col.className = "column " + (prob < 0.7 ? "green" : "red");
            col.innerHTML = `<h2>${sensor}</h2><p>${(prob*100).toFixed(1)}%</p>`;
            dashboard.appendChild(col);
          });
        }
        async function fetchPredictions() {
          try {
            const res = await fetch("/predictions");
            const data = await res.json();
            updateDashboard(data);
          } catch (error) {
            console.error("Ошибка получения предсказаний:", error);
          }
        }
        setInterval(fetchPredictions, 1000);
        fetchPredictions();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
