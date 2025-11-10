# =====================================
# 🌿 CLASIFICACIÓN DE HOJAS DE SOJA (TensorFlow CNN)
# =====================================

import os
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =====================================
# 1️⃣ CONFIGURACIÓN DE RUTAS
# =====================================
# ⚠️ CAMBIÁ ESTA RUTA a donde está tu carpeta "soybean"
BASE_PATH = r"/home/kmonti/Desktop/exagerada(640x360)/dataset_augmented-001/soybean"  # ⬅️ ejemplo Windows

# Carpeta donde se crearán los splits
SPLIT_PATH = os.path.join(os.path.dirname(BASE_PATH), "soybean_splits")
TRAIN_PATH = os.path.join(SPLIT_PATH, "train")
TEST_PATH = os.path.join(SPLIT_PATH, "test")

# Parámetros
TEST_SIZE = 0.2
RANDOM_STATE = 42
IMG_HEIGHT, IMG_WIDTH = 1920, 1080
BATCH_SIZE = 16
EPOCHS = 15

# =====================================
# 2️⃣ FUNCIÓN: CREAR TRAIN/TEST BALANCEADO
# =====================================
def crear_splits():
    print(f"\n📂 Leyendo clases desde: {BASE_PATH}")

    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"❌ No existe la carpeta base: {BASE_PATH}")

    clases = [c for c in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, c))]
    if not clases:
        raise RuntimeError("❌ No se encontraron subcarpetas de clases dentro de soybean/")
    print(f"✅ Clases detectadas: {clases}")

    # Borrar carpetas previas de splits
    for d in [TRAIN_PATH, TEST_PATH]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # Dividir cada clase
    for class_name in clases:
        class_path = os.path.join(BASE_PATH, class_name)
        images = [os.path.join(class_path, img)
                  for img in os.listdir(class_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) == 0:
            print(f"⚠️ Clase '{class_name}' no tiene imágenes, se omite.")
            continue

        if len(images) < 2:
            print(f"⚠️ Clase '{class_name}' tiene pocas imágenes ({len(images)}), todas irán a train.")
            train_imgs, test_imgs = images, []
        else:
            train_imgs, test_imgs = train_test_split(
                images, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

        os.makedirs(os.path.join(TRAIN_PATH, class_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_PATH, class_name), exist_ok=True)

        for img in train_imgs:
            shutil.copy(img, os.path.join(TRAIN_PATH, class_name))
        for img in test_imgs:
            shutil.copy(img, os.path.join(TEST_PATH, class_name))

        print(f"📸 Clase '{class_name}': {len(train_imgs)} train / {len(test_imgs)} test")

    print(f"\n✅ División completa:")
    print(f"   - Entrenamiento: {TRAIN_PATH}")
    print(f"   - Prueba: {TEST_PATH}\n")

# =====================================
# 3️⃣ FUNCIÓN: ENTRENAR CNN
# =====================================
def entrenar_modelo():
    print("🚀 Iniciando entrenamiento del modelo...\n")

    # Cargar datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"🧾 Clases utilizadas: {class_names}\n")

    # Optimización de rendimiento
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])

    normalization_layer = layers.Rescaling(1./255)
    num_classes = len(class_names)

    # Definición del modelo CNN
    model = models.Sequential([
        data_augmentation,
        normalization_layer,
        layers.Conv2D(32, (3, 3), activation='relu'),
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

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Entrenamiento
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS
    )

    # Evaluación
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n📊 Exactitud en test: {test_acc:.3f}")

    # Guardar modelo
    model_path = os.path.join(SPLIT_PATH, "modelo_soybean.h5")
    model.save(model_path)
    print(f"💾 Modelo guardado en: {model_path}")

    # Gráficos
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Entrenamiento')
    plt.plot(val_acc, label='Validación')
    plt.title('Exactitud')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Entrenamiento')
    plt.plot(val_loss, label='Validación')
    plt.title('Pérdida')
    plt.legend()
    plt.show()

# =====================================
# 4️⃣ BLOQUE PRINCIPAL
# =====================================
if __name__ == "__main__":
    print("🌱 --- MODELO CNN DE HOJAS DE SOJA ---")
    crear_splits()
    entrenar_modelo()
    print("\n✅ Proceso completado correctamente.")

