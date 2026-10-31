import os
import re
import joblib
import pymorphy3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Убираем пафосные названия — делаем простой студенческий заголовок
app = FastAPI(
    title="Курсовая работа: Классификатор писем",
    description="Простой сервис для проверки модели классификации служебных документов.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

morph = pymorphy3.MorphAnalyzer()

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model_pipeline.joblib")

try:
    pipeline = joblib.load(model_path)
    print("Модель загрузилась!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    pipeline = None


class Zapros(BaseModel):
    text: str = Field(..., title="Текст письма", example="Прошу дать отпуск на 14 дней")


class Otvet(BaseModel):
    status: str = Field(..., title="Статус")
    target_department: str = Field(..., title="Отдел")
    confidence: float = Field(..., title="Уверенность")
    probabilities: dict[str, float] = Field(..., title="Вероятности")


def preprocess_text(text: str) -> str:
    clean = re.sub(r'[^а-яё\s]', ' ', str(text).lower())
    words = clean.split()
    lemmas = [morph.parse(w)[0].normal_form for w in words if len(w) > 2]
    return ' '.join(lemmas)


# ------------------------------------------------------------------------------
# 1. Простая веб-страница
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home_page():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Классификатор документов</title>
        <style>
            body {
                background-color: #f0f0f0;
                color: #000000;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 25px 15px;
                display: flex;
                justify-content: center;
            }
            .main-box {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                width: 100%;
                max-width: 560px;
                padding: 20px;
            }
            h2 {
                margin-top: 0;
                font-size: 18px;
                color: #222222;
                border-bottom: 1px solid #eeeeee;
                padding-bottom: 8px;
            }
            p.info {
                font-size: 13px;
                color: #555555;
                margin-bottom: 15px;
            }
            label {
                font-size: 13px;
                font-weight: bold;
                display: block;
                margin-bottom: 5px;
            }
            textarea {
                width: 100%;
                box-sizing: border-box;
                height: 110px;
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #999999;
                border-radius: 3px;
                padding: 8px;
                font-size: 13px;
                font-family: inherit;
                outline: none;
            }
            textarea:focus {
                border-color: #0066cc;
            }
            .buttons-list {
                margin: 10px 0 15px 0;
            }
            .hint-btn {
                background-color: #e9e9e9;
                color: #333333;
                border: 1px solid #bbbbbb;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 12px;
                cursor: pointer;
                margin-right: 4px;
                margin-bottom: 5px;
            }
            .hint-btn:hover {
                background-color: #dddddd;
            }
            .submit-btn {
                width: 100%;
                background-color: #0066cc;
                color: #ffffff;
                border: 1px solid #0055aa;
                border-radius: 3px;
                padding: 10px;
                font-size: 14px;
                cursor: pointer;
            }
            .submit-btn:hover {
                background-color: #0055aa;
            }
            #result-block {
                display: none;
                margin-top: 18px;
                padding: 12px;
                background-color: #f9f9f9;
                border: 1px solid #dddddd;
                border-radius: 3px;
            }
            .res-header {
                font-size: 14px;
                margin-bottom: 8px;
            }
            .res-dept {
                font-weight: bold;
                color: #0055cc;
            }
            .status-tag {
                font-size: 12px;
                padding: 3px 6px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 10px;
                font-weight: bold;
            }
            .tag-green { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .tag-red { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            
            .row {
                display: flex;
                align-items: center;
                font-size: 12px;
                margin-bottom: 5px;
            }
            .row-title {
                width: 90px;
            }
            .bar-wrap {
                flex-grow: 1;
                background-color: #e5e5e5;
                height: 8px;
                border-radius: 2px;
                margin: 0 10px;
                overflow: hidden;
            }
            .bar-inner {
                height: 100%;
                background-color: #0066cc;
                width: 0%;
            }
            .row-val {
                width: 45px;
                text-align: right;
                color: #444444;
            }
        </style>
    </head>
    <body>
        <div class="main-box">
            <h2>Классификация служебных писем</h2>
            <p class="info">Вставьте текст обращения или выберите готовый пример для отправки в сервис.</p>
            
            <label for="text-field">Текст письма:</label>
            <textarea id="text-field" placeholder="Напишите текст..."></textarea>
            
            <div class="buttons-list">
                <button type="button" class="hint-btn" onclick="paste('Прошу предоставить очередной оплачиваемый отпуск на 14 дней с понедельника')">Отпуск (Кадры)</button>
                <button type="button" class="hint-btn" onclick="paste('Не работает локальная сеть и рабочий компьютер, почините пожалуйста')">Сбой сети (IT)</button>
                <button type="button" class="hint-btn" onclick="paste('Высылаем акт сверки расчетов и счет на оплату услуг')">Счет (Бухгалтерия)</button>
                <button type="button" class="hint-btn" onclick="paste('Прошу согласовать договор аренды помещения перед подписанием')">Договор (Юротдел)</button>
            </div>

            <button class="submit-btn" onclick="check()">Определить отдел</button>

            <div id="result-block">
                <div class="res-header">Назначенный отдел: <span class="res-dept" id="out-dept">—</span></div>
                <div id="out-status" class="status-tag">—</div>
                
                <div style="font-size: 12px; color: #666666; margin-bottom: 6px;">Распределение вероятностей:</div>
                <div id="out-bars"></div>
            </div>
        </div>

        <script>
            function paste(str) {
                const el = document.getElementById('text-field');
                el.value = str;
                el.focus();
            }

            async function check() {
                const val = document.getElementById('text-field').value.trim();
                if (!val) {
                    alert('Поле не должно быть пустым');
                    return;
                }

                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: val })
                });

                if (!res.ok) {
                    alert('Ошибка запроса к серверу');
                    return;
                }

                const data = await res.json();
                document.getElementById('result-block').style.display = 'block';
                document.getElementById('out-dept').innerText = data.target_department;

                const st = document.getElementById('out-status');
                if (data.status === 'AUTOMATIC') {
                    st.innerText = 'Автоматически (' + (data.confidence * 100).toFixed(1) + '%)';
                    st.className = 'status-tag tag-green';
                } else {
                    st.innerText = 'Ручная проверка оператором (' + (data.confidence * 100).toFixed(1) + '%)';
                    st.className = 'status-tag tag-red';
                }

                const container = document.getElementById('out-bars');
                container.innerHTML = '';
                for (const [dept, prob] of Object.entries(data.probabilities)) {
                    const percent = (prob * 100).toFixed(1);
                    container.innerHTML += `
                        <div class="row">
                            <span class="row-title">${dept}</span>
                            <div class="bar-wrap"><div class="bar-inner" style="width: ${percent}%;"></div></div>
                            <span class="row-val">${percent}%</span>
                        </div>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """


# ------------------------------------------------------------------------------
# 2. Простые эндпоинты для Swagger (/docs)
# ------------------------------------------------------------------------------
@app.get("/health", tags=["Проверка работы"], summary="Проверка сервера")
def health_check():
    """Проверяем, что файл модели загружен в память."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=Otvet, tags=["Классификация"], summary="Определить отдел по тексту")
def route_document(request: Zapros):
    """Отправляем текст письма, получаем отдел и вероятности."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не найдена")

    cleaned_input = preprocess_text(request.text)
    if not cleaned_input:
        raise HTTPException(status_code=400, detail="Текст пустой после очистки")

    probabilities = pipeline.predict_proba([cleaned_input])[0]
    classes = pipeline.classes_

    best_idx = probabilities.argmax()
    best_dept = classes[best_idx]
    confidence = float(probabilities[best_idx])

    # Порог уверенности 60%
    if confidence >= 0.60:
        routing_status = "AUTOMATIC"
        target_dept = best_dept
    else:
        routing_status = "MANUAL_REVIEW_REQUIRED"
        target_dept = "Ручная проверка (Оператор)"

    prob_map = {cls: round(float(p), 4) for cls, p in zip(classes, probabilities)}

    return Otvet(
        status=routing_status,
        target_department=target_dept,
        confidence=round(confidence, 4),
        probabilities=prob_map
    )
