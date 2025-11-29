import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

# 1. НАСТРОЙКИ


IMG_SIZE = (200, 200)
BATCH_SIZE = 16
EPOCHS = 30
SEED = 42



DATA_DIR_TRAIN = r"C:/Users/Kuat\Documents/AI engineering course/cv-defect-classifier/NEU_surface_defects/train/images"
DATA_DIR_VAL   = r"C:/Users/Kuat/Documents/AI engineering course/cv-defect-classifier/NEU_surface_defects/validation/images"


assert os.path.isdir(DATA_DIR_TRAIN), f"Train dir not found: {DATA_DIR_TRAIN}"
assert os.path.isdir(DATA_DIR_VAL), f"Val dir not found: {DATA_DIR_VAL}"

print("Train dir:", DATA_DIR_TRAIN)
print("Val dir:  ", DATA_DIR_VAL)


# 2. ЗАГРУЗКА ДАННЫХ


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
)

train_data = train_datagen.flow_from_directory(
    directory=DATA_DIR_TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED,
)

val_data = val_datagen.flow_from_directory(
    directory=DATA_DIR_VAL,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

num_classes = train_data.num_classes
class_indices = train_data.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(num_classes)]

print("Классы (по индексам):", class_names)


# 3. МОДЕЛЬ ResNet50V2

base_model = tf.keras.applications.ResNet50V2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet',
)


base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = inputs


x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs, name="neu_resnet50v2")
model.summary()


# 4. План обучения — Adam + LearningRateScheduler


initial_lr = 1e-4

def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        new_lr = lr * math.exp(-0.1)  
        return float(new_lr)


lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "neu_best_resnet50v2.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

earlystop_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# 5. ОБУЧЕНИЕ


history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[lr_callback, checkpoint_cb, earlystop_cb],
    verbose=1,
)


# 6. ГРАФИКИ ОБУЧЕНИЯ


def plot_history(history):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.show()

plot_history(history)


# 7. ОЦЕНКА НА ВАЛИДАЦИИ


val_data.reset()
pred_probs = model.predict(val_data, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_data.classes
target_names = class_names

print("\nClassification report (VAL):")
print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion matrix (VAL)")
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, ha="right")
plt.yticks(tick_marks, target_names)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()


# 8. СОХРАНЕНИЕ МОДЕЛИ

model.save("neu_best_finetuned.keras")

with open("class_names.txt", "w", encoding="utf-8") as f:
    for name in target_names:
        f.write(name + "\n")

print("Модель и список классов сохранены:")
print("  neu_best_finetuned.keras")
print("  class_names.txt")
