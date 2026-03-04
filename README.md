# CV Defect Classifier

Простой проект для классификации дефектов поверхности металла по изображению.

Используется:
- Flask (веб-интерфейс + API)
- TensorFlow/Keras (модель классификации)
- датасет NEU Surface Defects

## Что умеет

- загрузка изображения через веб-страницу
- предсказание класса дефекта
- вывод уверенности и вероятностей по классам
- запуск в Docker

## Требования

- Python 3.11+
- зависимости из `requirements.txt`

## Установка

```powershell
cd c:\Users\Documents\CV-Defect-Classifie
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

## Данные

Скачайте NEU Surface Defects и разложите в `datasets/`.

Минимально нужная структура:

```text
datasets/
  train/
    images/
      crazing/
      inclusion/
      patches/
      pitted_surface/
      rolled-in_scale/
      scratches/
  validation/
    images/
      crazing/
      inclusion/
      patches/
      pitted_surface/
      rolled-in_scale/
      scratches/
```

## Обучение модели

```powershell
python -m training.train_neu_model
```

После обучения сохраняются:
- модель: `app/models/neu_best_finetuned.keras`
- классы: `app/models/class_names.txt`
- метрики и графики: `results/`

## Запуск приложения

```powershell
python -m flask run
```

или:

```powershell
$env:FLASK_APP = "app.app"
flask run
```

Адрес: `http://localhost:5000`

## API

Эндпоинт:

```text
POST /api/predict
```

Пример запроса:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

Пример ответа:

```json
{
  "predicted_class": "scratches",
  "confidence": 0.95,
  "probabilities": {
    "crazing": 0.01,
    "inclusion": 0.01,
    "patches": 0.01,
    "pitted_surface": 0.01,
    "rolled-in_scale": 0.01,
    "scratches": 0.95
  }
}
```

## Grad-CAM (пояснение предсказания)

```powershell
python -m training.grad_cam_demo
```

Скрипт показывает:
- исходное изображение
- heatmap
- наложение heatmap на изображение

## Docker

```bash
docker build -t cv-defect-classifier .
docker run -p 5000:5000 cv-defect-classifier
```

## Классы дефектов

- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches
