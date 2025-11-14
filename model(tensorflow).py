# =====================================
# 🌿 CLASIFICACIÓN DE HOJAS DE SOJA (TensorFlow CNN)
# =====================================

import os
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json
import stat
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from datetime import datetime

# =====================================
# 1️⃣ CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# =====================================
BASE_PATH = r"C:\Users\neiel\OneDrive\Desktop\exagerada(640x360)\dataset_augmented-001\soybean"

SPLIT_PATH = os.path.join(os.path.dirname(BASE_PATH), "soybean_splits")
TRAIN_PATH = os.path.join(SPLIT_PATH, "train")
TEST_PATH = os.path.join(SPLIT_PATH, "test")
RESULTS_LOG = os.path.join(SPLIT_PATH, "resultados_modelos.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
IMG_HEIGHT, IMG_WIDTH = 360, 640
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.01

# =====================================
# 2️⃣ FUNCIONES AUXILIARES
# =====================================
def force_remove_readonly(func, path, excinfo):
    """Evita errores de permisos (WinError 5) al borrar carpetas en Windows."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def crear_splits():
    print(f"\n📂 Leyendo clases desde: {BASE_PATH}")

    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"❌ No existe la carpeta base: {BASE_PATH}")

    clases = [c for c in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, c))]
    if not clases:
        raise RuntimeError("❌ No se encontraron subcarpetas de clases dentro de soybean/")
    print(f"✅ Clases detectadas: {clases}")

    # Borrar carpetas previas
    for d in [TRAIN_PATH, TEST_PATH]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d, onerror=force_remove_readonly)
            except PermissionError as e:
                print(f"⚠️ No se pudo borrar {d} ({e}), se continuará de todos modos.")
        os.makedirs(d, exist_ok=True)

    for class_name in clases:
        class_path = os.path.join(BASE_PATH, class_name)
        images = [os.path.join(class_path, img)
                  for img in os.listdir(class_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) == 0:
            print(f"⚠️ Clase '{class_name}' vacía, se omite.")
            continue

        if len(images) < 2:
            train_imgs, test_imgs = images, []
        else:
            train_imgs, test_imgs = train_test_split(
                images, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

        for split, imgs in [('train', train_imgs), ('test', test_imgs)]:
            target_dir = os.path.join(SPLIT_PATH, split, class_name)
            os.makedirs(target_dir, exist_ok=True)
            for img in imgs:
                shutil.copy(img, target_dir)

        print(f"📸 {class_name}: {len(train_imgs)} train / {len(test_imgs)} test")

    print("\n✅ División completa.\n")
    return clases

# =====================================
# 3️⃣ ENTRENAMIENTO DEL MODELO
# =====================================
def entrenar_modelo():
    print("🚀 Iniciando entrenamiento del modelo...\n")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False  # 🔧 Necesario para matriz de confusión
    )

    class_names = train_ds.class_names
    print(f"🧾 Clases: {class_names}\n")

    # Guardar clases
    with open(os.path.join(SPLIT_PATH, "clases_soybean.json"), "w") as f:
        json.dump(class_names, f)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])

    normalization_layer = layers.Rescaling(1./255)
    num_classes = len(class_names)

    model = models.Sequential([
        data_augmentation,
        normalization_layer,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n📊 Exactitud global: {test_acc:.3f}")

    # Guardar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"modelo_soybean_{timestamp}.h5"
    model_path = os.path.join(SPLIT_PATH, model_name)
    model.save(model_path)

    # =====================================
    # 🔍 EVALUACIÓN DETALLADA
    # =====================================
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\n📋 Reporte de clasificación por clase:")
    print(report_df)

    # Guardar matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    cm_path = os.path.join(SPLIT_PATH, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # Guardar reporte CSV
    report_path = os.path.join(SPLIT_PATH, f"reporte_{timestamp}.csv")
    report_df.to_csv(report_path)

    # Guardar log resumen (para comparar modelos)
    resumen = {
        "fecha": timestamp,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "accuracy": test_acc,
        "f1_macro": report_df.loc["macro avg", "f1-score"],
        "modelo": model_name,
        "conf_matrix": os.path.basename(cm_path),
    }

    resumen_df = pd.DataFrame([resumen])
    if os.path.exists(RESULTS_LOG):
        resumen_df.to_csv(RESULTS_LOG, mode='a', header=False, index=False)
    else:
        resumen_df.to_csv(RESULTS_LOG, index=False)

    print(f"\n🧾 Reporte detallado guardado en: {report_path}")
    print(f"💾 Resultados agregados a: {RESULTS_LOG}")
    print(f"🧠 Modelo guardado en: {model_path}")
    print(f"📉 Matriz de confusión: {cm_path}")

# =====================================
# 4️⃣ MAIN
# =====================================
if __name__ == "__main__":
    print("🌱 --- MODELO CNN DE HOJAS DE SOJA ---")
    crear_splits()
    entrenar_modelo()
    print("\n✅ Proceso completado correctamente.")

