import os
import random
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# ============================================
# CONFIGURACIÓN: EDITAR ESTAS RUTAS
# ============================================
PATH_SOJA = r"/home/kmonti/Desktop/soja"
PATH_NOSOJA = r"/home/kmonti/Desktop/no_soja"    # YA MEZCLADO

# Cantidad máxima de NO-SOJA a usar (ratio 1:4 → = 4 × 593)
RATIO = 4

# Salida
OUTPUT = r"/home/kmonti/Desktop/dataset_binario"

# Tamaño test
TEST_SIZE = 0.20
SEED = 42

# ============================================
# FUNCIONES
# ============================================

def recolectar_imagenes(path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")
    imgs = []
    for e in exts:
        imgs.extend(glob(os.path.join(path, "**", e), recursive=True))
    return imgs


def copiar_archivos(lista_paths, destino):
    os.makedirs(destino, exist_ok=True)
    for src in lista_paths:
        nombre = os.path.basename(src)
        dst = os.path.join(destino, nombre)
        shutil.copy2(src, dst)


# ============================================
# PASO 1 — Recolectar imágenes
# ============================================
print("Recolectando imágenes...")

soja_imgs = recolectar_imagenes(PATH_SOJA)
nosoja_imgs = recolectar_imagenes(PATH_NOSOJA)

print(f"✔ Soja: {len(soja_imgs)} imágenes")
print(f"✔ No-soja (total): {len(nosoja_imgs)} imágenes")

# ============================================
# PASO 2 — Undersampling de no-soja
# ============================================
N_SOJA = len(soja_imgs)
N_NOSOJA_TARGET = N_SOJA * RATIO

print(f"Seleccionando aleatoriamente {N_NOSOJA_TARGET} imágenes no-soja...")

random.shuffle(nosoja_imgs)
nosoja_final = nosoja_imgs[:N_NOSOJA_TARGET]

print(f"✔ Seleccionadas: {len(nosoja_final)} imágenes no-soja")

# ============================================
# PASO 3 — Armar dataset binario
# ============================================
X = soja_imgs + nosoja_final
y = [0] * len(soja_imgs) + [1] * len(nosoja_final)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=y
)

# ============================================
# PASO 4 — Copiar archivos a carpetas finales
# ============================================
dest_train_soja = os.path.join(OUTPUT, "train", "soja")
dest_train_nosoja = os.path.join(OUTPUT, "train", "nosoja")
dest_test_soja = os.path.join(OUTPUT, "test", "soja")
dest_test_nosoja = os.path.join(OUTPUT, "test", "nosoja")

print("Copiando archivos...")

for path, label in zip(train_paths, train_labels):
    destino = dest_train_soja if label == 0 else dest_train_nosoja
    copiar_archivos([path], destino)

for path, label in zip(test_paths, test_labels):
    destino = dest_test_soja if label == 0 else dest_test_nosoja
    copiar_archivos([path], destino)

print("==============================================")
print("✔ Dataset binario balanceado generado con éxito")
print(f"Ruta: {OUTPUT}")
print("==============================================")