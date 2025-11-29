

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image


tf.config.run_functions_eagerly(True)



IMG_SIZE = (200, 200)  


MODEL_PATH = r"C:\Users\Kuat\Documents\AI engineering course\cv-defect-classifier\neu_best_resnet50v2.keras"
CLASS_NAMES_PATH = r"C:\Users\Kuat\Documents\AI engineering course\cv-defect-classifier\class_names.txt"


IMG_PATH = r"C:\Users\Kuat\Documents\AI engineering course\cv-defect-classifier\NEU_surface_defects\validation\images\inclusion\inclusion_255.jpg"

print("MODEL_PATH:", MODEL_PATH)
print("CLASS_NAMES_PATH:", CLASS_NAMES_PATH)
print("IMG_PATH:", IMG_PATH)



if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути:\n{MODEL_PATH}")

if not os.path.isfile(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"class_names.txt не найден по пути:\n{CLASS_NAMES_PATH}")

if not os.path.isfile(IMG_PATH):
    raise FileNotFoundError(f"Изображение не найдено по пути:\n{IMG_PATH}")



model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

print("Классы модели:", class_names)


backbone = model.get_layer("resnet50v2")
print("Backbone name:", backbone.name)
print("Backbone output shape:", backbone.output_shape)




feature_input = tf.keras.Input(shape=backbone.output_shape[1:])


x = feature_input
apply = False
for layer in model.layers:
    
    if layer.name == backbone.name:
        apply = True
        continue
    if apply:
        
        x = layer(x)

head_model = tf.keras.Model(feature_input, x, name="head_model")

print("Head model summary:")
head_model.summary()



def load_and_preprocess_image(img_path, img_size=IMG_SIZE):
    
    img = keras_image.load_img(img_path, target_size=img_size)
    img_array = keras_image.img_to_array(img).astype("float32") / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img_array  


def make_gradcam_heatmap(img_array_batch):
    
    
    with tf.GradientTape() as tape:
        
        feature_maps = backbone(img_array_batch, training=False)  

        
        tape.watch(feature_maps)

        
        predictions = head_model(feature_maps, training=False)  

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]  

    
    grads = tape.gradient(class_channel, feature_maps)  

    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  

    feature_maps = feature_maps[0]  

    
    heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)  

    
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()


def show_gradcam(img_path):
    
    print(f"\nОбработка изображения: {img_path}")
    img_batch, img_array = load_and_preprocess_image(img_path)

    heatmap, preds = make_gradcam_heatmap(img_batch)

    pred_index = int(np.argmax(preds[0]))
    pred_class = class_names[pred_index]
    confidence = float(preds[0][pred_index])

    
    h, w, _ = img_array.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    
    original_uint8 = (img_array * 255).astype("uint8")

    superimposed_img = heatmap_color * 0.4 + original_uint8
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_uint8)
    plt.title("Исходное изображение")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f"{pred_class} ({confidence*100:.2f}%)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Предсказанный класс: {pred_class}, уверенность: {confidence*100:.2f}%")


if __name__ == "__main__":
    show_gradcam(IMG_PATH)
