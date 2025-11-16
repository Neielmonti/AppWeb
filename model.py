# =====================================
# 🌿 CLASIFICACIÓN DE HOJAS DE SOJA CON TRANSFER LEARNING (ImageNet)
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
# 1️⃣ CONFIGURACIÓN
# =====================================
BASE_PATH = r"C:\Users\neiel\OneDrive\Desktop\dataset_preprocesado"

SPLIT_PATH = os.path.join(os.path.dirname(BASE_PATH), "soybean_splits")
TRAIN_PATH = os.path.join(SPLIT_PATH, "train")
TEST_PATH = os.path.join(SPLIT_PATH, "test")
RESULTS_LOG = os.path.join(SPLIT_PATH, "resultados_modelos.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
EPOCHS = 70
LEARNING_RATE = 0.00001

# =====================================
# 2️⃣ FUNCIONES AUXILIARES
# =====================================
def force_remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# 🔹 NUEVO ───────────────────────────────────────────────
def es_imagen_valida(path):
    """Devuelve True si la imagen puede cargarse, False si está corrupta."""
    try:
        img = tf.io.read_file(path)
        _ = tf.image.decode_image(img)  # fuerza a TF a decodificar
        return True
    except Exception as e:
        print(f"⚠️ Archivo corrupto ignorado: {path}")
        return False
# ─────────────────────────────────────────────────────────

def crear_splits():
    print(f"\n📂 Leyendo clases desde: {BASE_PATH}")

    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"❌ No existe la carpeta base: {BASE_PATH}")

    clases = [c for c in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, c))]
    print(f"✅ Clases detectadas: {clases}")

    for d in [TRAIN_PATH, TEST_PATH]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d, onerror=force_remove_readonly)
            except PermissionError as e:
                print(f"⚠️ No se pudo borrar {d}: {e}")
        os.makedirs(d, exist_ok=True)

    for class_name in clases:
        class_path = os.path.join(BASE_PATH, class_name)

        # 🔹 NUEVO: filtro de imágenes corruptas
        images = []
        for img in os.listdir(class_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(class_path, img)
                if es_imagen_valida(full_path):
                    images.append(full_path)

        if not images:
            continue

        train_imgs, test_imgs = train_test_split(images, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        for split, imgs in [('train', train_imgs), ('test', test_imgs)]:
            target_dir = os.path.join(SPLIT_PATH, split, class_name)
            os.makedirs(target_dir, exist_ok=True)
            for img in imgs:
                shutil.copy(img, target_dir)

    print("\n✅ División completa.\n")
    return clases

# === Callbacks simples y potentes ===
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

checkpoint_path = os.path.join(SPLIT_PATH, "best_model.keras")

callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]

# =====================================
# 3️⃣ ENTRENAMIENTO CON TRANSFER LEARNING
# =====================================
def entrenar_modelo():
    print("🚀 Iniciando entrenamiento con modelo preentrenado (ImageNet)...\n")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"🧾 Clases: {class_names}\n")

    with open(os.path.join(SPLIT_PATH, "clases_soybean.json"), "w") as f:
        json.dump(class_names, f)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    """

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        name=None,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    # x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    print("CANTIDAD DE PESOS GUARDADOS:", len(base_model.get_weights()))
    print("Primeros 5 pesos (std):")
    for w in base_model.get_weights()[:5]:
        print(w.std())

    #history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n📊 Exactitud global: {test_acc:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"modelo_soybean_imagenet_{timestamp}.h5"
    model_path = os.path.join(SPLIT_PATH, model_name)
    model.save(model_path.replace(".h5", ".keras"), save_format="keras")

    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\n📋 Reporte de clasificación por clase:")
    print(report_df)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de confusión (ImageNet)")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    cm_path = os.path.join(SPLIT_PATH, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    report_path = os.path.join(SPLIT_PATH, f"reporte_{timestamp}.csv")
    report_df.to_csv(report_path)

    resumen = {
        "fecha": timestamp,
        "modelo": "MobileNetV2",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "accuracy": test_acc,
        "f1_macro": report_df.loc["macro avg", "f1-score"],
        "modelo_path": model_name,
        "conf_matrix": os.path.basename(cm_path),
    }
    resumen_df = pd.DataFrame([resumen])
    if os.path.exists(RESULTS_LOG):
        resumen_df.to_csv(RESULTS_LOG, mode='a', header=False, index=False)
    else:
        resumen_df.to_csv(RESULTS_LOG, index=False)

    print(f"\n🧾 Reporte guardado en: {report_path}")
    print(f"💾 Log agregado a: {RESULTS_LOG}")
    print(f"🧠 Modelo guardado en: {model_path}")

# =====================================
# 4️⃣ MAIN
# =====================================
if __name__ == "__main__":
    print("🌱 --- MODELO CNN DE HOJAS DE SOJA (ImageNet) ---")
    crear_splits()
    entrenar_modelo()
    print("\n✅ Proceso completado correctamente.")

