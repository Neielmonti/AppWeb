import cv2
import numpy as np
import os

# === PARÁMETROS ===
FH = 1.4
FS = 1.5
FB = 1.2
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}

def procesar_imagen(ruta_entrada, ruta_salida):
    img = cv2.imread(ruta_entrada)
    if img is None:
        print(f"⚠️ No se pudo cargar la imagen: {ruta_entrada}")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")

    h, s, v = cv2.split(hsv)
    h_mean = np.mean(h)
    h_mod = np.clip((h - h_mean) * FH + h_mean, 0, 179)
    s_mod = np.clip(s * FS, 0, 255)
    v_mod = np.clip(v * FB, 0, 255)

    hsv_mod = cv2.merge([h_mod, s_mod, v_mod])
    img_mod = cv2.cvtColor(hsv_mod.astype("uint8"), cv2.COLOR_HSV2BGR)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    cv2.imwrite(ruta_salida, img_mod)
    return True


def exagerar_dataset(carpeta_entrada, carpeta_salida):
    procesadas = []
    for root, _, files in os.walk(carpeta_entrada):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXT:
                ruta_in = os.path.join(root, file)
                ruta_rel = os.path.relpath(ruta_in, carpeta_entrada)
                ruta_out = os.path.join(carpeta_salida, ruta_rel)

                if procesar_imagen(ruta_in, ruta_out):
                    procesadas.append(ruta_rel)
    return procesadas