import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from datetime import datetime

# ============================================
# CONFIGURACIÓN
# ============================================
DATASET = r"C:\Users\neiel\OneDrive\Desktop\dataset_binario"
IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 25
LR = 1e-4

OUTPUT_DIR = os.path.join(DATASET, "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# FUNCIÓN PARA FILTRAR IMÁGENES CORRUPTAS
# ============================================
def es_imagen_valida(path):
    """Devuelve True si la imagen puede cargarse, False si está corrupta."""
    try:
        raw = tf.io.read_file(path)
        _ = tf.image.decode_image(raw)
        return True
    except Exception:
        print(f"⚠️ Archivo corrupto ignorado: {path}")
        return False


def limpiar_directorio(directorio):
    """Elimina físicamente las imágenes corruptas del dataset."""
    count = 0
    for root, _, files in os.walk(directorio):
        for f in files:
            path = os.path.join(root, f)
            if not es_imagen_valida(path):
                os.remove(path)
                count += 1
    print(f"✔ Limpieza completada en {directorio}. Eliminados: {count} archivos corruptos.")


# ============================================
# LIMPIAR TRAIN Y TEST ANTES DE CARGARLOS
# ============================================
print("🔍 Escaneando imágenes corruptas...")
limpiar_directorio(os.path.join(DATASET, "train"))
limpiar_directorio(os.path.join(DATASET, "test"))
print("🔍 Escaneo finalizado.\n")


# ============================================
# CARGAR DATASET (YA SIN IMÁGENES CORRUPTAS)
# ============================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False
)

class_names = train_ds.class_names
print("Clases:", class_names)

with open(os.path.join(OUTPUT_DIR, "clases_binario.json"), "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(300).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)


# ============================================
# MODELO BINARIO — TRANSFER LEARNING
# ============================================
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================
# ENTRENAMIENTO
# ============================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# ============================================
# CURVAS
# ============================================
plt.plot(history.history["accuracy"], label="acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Accuracy binario")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_binario.png"))
plt.close()


# ============================================
# EVALUACIÓN
# ============================================
y_true = np.concatenate([y for _, y in val_ds])
y_pred_prob = model.predict(val_ds)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\n=== REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names))

report_df = pd.DataFrame(
    classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
)
report_df.to_csv(os.path.join(OUTPUT_DIR, "reporte_binario.csv"))


# ============================================
# MATRIZ DE CONFUSIÓN
# ============================================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=class_names, yticklabels=class_names,
    cmap="Blues"
)
plt.title("Matriz de confusión binaria")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_binaria.png"))
plt.close()


# ============================================
# GUARDAR MODELO
# ============================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path_model = os.path.join(OUTPUT_DIR, f"modelo_binario_{ts}.keras")
model.save(path_model)

print("\n✔ MODELO BINARIO GUARDADO EN:")
print(path_model)

