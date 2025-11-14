# pipeline.py
import cv2
import os
import uuid
from estandarizar_tamano import estandarizar_imagen
from exageracion_hsv import exagerar_imagen, procesar_iluminacion

def procesar_imagen_pipeline(ruta_in, ruta_out):
    """
    Aplica el pipeline completo:
    1) corregir iluminación
    2) estandarizar tamaño (640×360)
    3) exagerar color (HSV)
    Guarda la imagen final en ruta_out.
    """
    temp1 = ruta_out.replace(".png", "_corr.png")
    temp2 = ruta_out.replace(".png", "_std.png")

    # 1) leer img original
    img = cv2.imread(ruta_in)
    if img is None:
        print("No se pudo leer:", ruta_in)
        return False

    # 1) iluminación
    img_corr = procesar_iluminacion(img)
    cv2.imwrite(temp1, img_corr)

    # 2) estandarizar (usa temp1 → temp2)
    estandarizar_imagen(temp1, temp2)

    # 3) exagerar (usa temp2 → ruta final)
    exagerar_imagen(temp2, ruta_out)

    # limpiar temporales
    try:
        os.remove(temp1)
        os.remove(temp2)
    except:
        pass

    return True