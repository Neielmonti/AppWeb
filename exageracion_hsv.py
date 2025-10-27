import cv2
import numpy as np
import os

# === PARÁMETROS POR DEFECTO ===
FH = 1.4
FS = 1.5
FB = 1.2
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}


def exagerar_imagen(ruta_entrada, ruta_salida, fh=FH, fs=FS, fb=FB):
    """
    Aumenta la saturación, brillo y contraste de tono de una imagen.
    Mantiene el formato sin deformar ni cambiar tamaño.
    """
    img = cv2.imread(ruta_entrada)
    if img is None:
        print(f"⚠️ No se pudo cargar la imagen: {ruta_entrada}")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")

    h, s, v = cv2.split(hsv)
    h_mean = np.mean(h)
    h_mod = np.clip((h - h_mean) * fh + h_mean, 0, 179)
    s_mod = np.clip(s * fs, 0, 255)
    v_mod = np.clip(v * fb, 0, 255)

    hsv_mod = cv2.merge([h_mod, s_mod, v_mod])
    img_mod = cv2.cvtColor(hsv_mod.astype("uint8"), cv2.COLOR_HSV2BGR)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    cv2.imwrite(ruta_salida, img_mod)
    print(f"✅ Imagen exagerada guardada en {ruta_salida}")
    return True


def exagerar_dataset(carpeta_entrada, carpeta_salida, fh=FH, fs=FS, fb=FB, verbose=True):
    """
    Recorre una estructura de carpetas y exagera los colores de todas las imágenes.
    Mantiene la estructura original.
    """
    procesadas = []
    count = 0

    for root, _, files in os.walk(carpeta_entrada):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXT:
                ruta_in = os.path.join(root, file)
                ruta_rel = os.path.relpath(ruta_in, carpeta_entrada)
                ruta_out = os.path.join(carpeta_salida, ruta_rel)

                if exagerar_imagen(ruta_in, ruta_out, fh, fs, fb):
                    procesadas.append(ruta_rel)
                    count += 1
                    if verbose and count % 20 == 0:
                        print(f"📸 {count} imágenes procesadas...")

    print(f"✅ Total de imágenes procesadas: {count}")
    return procesadas

def procesar_iluminacion(img):
    # Convertimos a espacio LAB (mejor para iluminación)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Aplicamos CLAHE pero con parámetros suaves
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)

    # Recombinar canales
    lab_eq = cv2.merge((l_eq, a, b))
    corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Convertir a float para ajustes finos
    corrected = corrected.astype(np.float32) / 255.0

    # Reducir un poco el brillo global si se pasa
    corrected = np.clip(corrected * 0.9, 0, 1)

    # Suavizar contraste global (para evitar sobreexposición)
    corrected = cv2.pow(corrected, 1.05)

    corrected = (corrected * 255).astype(np.uint8)
    return corrected