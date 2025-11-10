# estandarizar_tamano.py
import cv2
import os
import traceback

# Extensiones válidas de imagen
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}

# Tamaño objetivo pensado para fotos de celular
# Mantiene relación 16:9, pero más liviano
TAM_OBJ = (640, 360)  # (ancho, alto)

def estandarizar_imagen(ruta_entrada, ruta_salida, tamaño=TAM_OBJ):
    """
    Redimensiona y recorta centrado una imagen para ajustarla al tamaño objetivo.
    Si la imagen no puede cargarse o procesarse, devuelve False.
    """
    try:
        img = cv2.imread(ruta_entrada)
        if img is None:
            print(f"⚠️ No se pudo leer la imagen (posiblemente dañada): {ruta_entrada}")
            return False

        h, w = img.shape[:2]
        target_w, target_h = tamaño

        # Escalado proporcional
        escala = max(target_w / w, target_h / h)
        nuevo_w = int(w * escala)
        nuevo_h = int(h * escala)

        img_redim = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)

        # Recorte centrado
        start_x = (nuevo_w - target_w) // 2
        start_y = (nuevo_h - target_h) // 2
        img_final = img_redim[start_y:start_y + target_h, start_x:start_x + target_w]

        # Guardar en carpeta destino
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        cv2.imwrite(ruta_salida, img_final)
        return True

    except Exception as e:
        print(f"❌ Error procesando {ruta_entrada}: {e}")
        traceback.print_exc(limit=1)
        return False


def estandarizar_dataset(carpeta_entrada, carpeta_salida, tamaño=TAM_OBJ, paso=100):
    """
    Recorre toda la carpeta con subdirectorios y estandariza todas las imágenes válidas.
    Ignora archivos dañados o ilegibles.
    Muestra progreso cada 'paso' imágenes.
    """
    procesadas = []
    contador = 0
    errores = 0

    # Calcular total estimado de imágenes válidas
    total = sum(
        len([f for f in files if os.path.splitext(f)[1].lower() in ALLOWED_EXT])
        for _, _, files in os.walk(carpeta_entrada)
    )

    print(f"📂 Iniciando estandarización de {total} imágenes desde '{carpeta_entrada}'...\n")

    for root, _, files in os.walk(carpeta_entrada):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in ALLOWED_EXT:
                continue

            ruta_in = os.path.join(root, file)
            ruta_rel = os.path.relpath(ruta_in, carpeta_entrada)
            ruta_out = os.path.join(carpeta_salida, ruta_rel)

            ok = estandarizar_imagen(ruta_in, ruta_out, tamaño=tamaño)
            if ok:
                procesadas.append(ruta_rel)
            else:
                errores += 1

            contador += 1
            if contador % paso == 0 or contador == total:
                print(f"🔹 Procesadas {contador}/{total} imágenes... ({errores} errores acumulados)")

    print(f"\n✅ Procesamiento completo. Total exitosas: {len(procesadas)}, errores: {errores}")
    return procesadas


if __name__ == "__main__":
    # Ejemplo de uso directo
    entrada = r"D:\ruta\a\dataset_original"
    salida = r"D:\ruta\a\dataset_estandarizado"
    estandarizar_dataset(entrada, salida, tamaño=TAM_OBJ, paso=200)