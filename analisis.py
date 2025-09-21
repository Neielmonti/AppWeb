import os
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import uuid


def analizar_imagen(ruta_imagen, carpeta_resultados):
    img = Image.open(ruta_imagen)
    img_array = np.array(img)

    ancho, alto = img.size
    formato = img.format
    modo = img.mode
    canales = len(modo)
    bits_por_canal = img.bits
    bits_por_pixel = bits_por_canal * canales

    try:
        n_frames = img.n_frames
    except AttributeError:
        n_frames = 1

    # Escala de grises para histogramas
    if img_array.ndim == 3:
        gris = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gris = img_array

    valor_min = int(img_array.min())
    valor_max = int(img_array.max())
    rango_dinamico = valor_max - valor_min
    media = float(np.mean(img_array))
    desvio = float(np.std(img_array))
    moda = int(Counter(gris.flatten()).most_common(1)[0][0])

    # Guardar histograma en archivo
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(gris.flatten(), bins=256, color="black", alpha=0.7)
    axs[0].set_title("Histograma (gris)")

    if img_array.ndim == 3 and img_array.shape[2] == 3:
        colores = ["r", "g", "b"]
        etiquetas = ["Rojo", "Verde", "Azul"]
        for i, (c, label) in enumerate(zip(colores, etiquetas)):
            axs[1].hist(img_array[:, :, i].flatten(), bins=256, color=c, alpha=0.5, label=label)
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "No es RGB", ha="center", va="center")

    hist_name = f"hist_{uuid.uuid4().hex}.png"
    hist_path = os.path.join(carpeta_resultados, hist_name)
    plt.savefig(hist_path)
    plt.close(fig)

    stats = {
        "formato": formato,
        "ancho": ancho,
        "alto": alto,
        "modo": modo,
        "bits_por_pixel": bits_por_pixel,
        "canales": canales,
        "frames": n_frames,
        "rango_dinamico": rango_dinamico,
        "min": valor_min,
        "max": valor_max,
        "media": round(media, 2),
        "desvio": round(desvio, 2),
        "moda": moda
    }

    return stats, hist_path.replace("\\", "/")  # ruta usable en HTML


# --- Función para analizar carpeta ---
def analizar_carpeta(ruta_carpeta, mostrar_histograma=False):
    extensiones_validas = [".jpg", ".jpeg", ".png", ".tif", ".bmp"]
    archivos = [f for f in os.listdir(ruta_carpeta) if os.path.splitext(f)[1].lower() in extensiones_validas]
    print(f"🔎 Encontradas {len(archivos)} imágenes en la carpeta '{ruta_carpeta}'")
    for archivo in archivos:
        ruta_imagen = os.path.join(ruta_carpeta, archivo)
        analizar_imagen(ruta_imagen, mostrar_histograma)

ruta_carpeta = "plant_classes"

#analizar_carpeta(ruta_carpeta, mostrar_histograma=True)

if __name__ == "__main__":
    ruta_carpeta = "plant_classes"
    analizar_carpeta(ruta_carpeta, mostrar_histograma=True)