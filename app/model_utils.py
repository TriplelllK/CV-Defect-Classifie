# model util
import os
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "neu_best_finetuned.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.txt")

IMG_SIZE = (200, 200)  # same as train

# load model once
model = tf.keras.models.load_model(MODEL_PATH)

# load names
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]


def preprocess_image(file) -> np.ndarray:
    """
    file: werkzeug FileStorage (request.files['file'])
    Возвращает np.array формы (1, 200, 200, 3) с нормировкой 0..1,
    как при обучении (ImageDataGenerator(rescale=1/255)).
    """
    file.stream.seek(0)

    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype("float32") / 255.0  # norm
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def predict_defect(file):
    """
    Принимает загруженный файл, возвращает (pred_class, confidence, probs_dict)
    """
    img_batch = preprocess_image(file)
    preds = model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = class_names[pred_idx]
    confidence = float(preds[pred_idx])

    probs_dict = {
        class_names[i]: float(preds[i])
        for i in range(len(class_names))
    }

    return pred_class, confidence, probs_dict
