from PIL import Image, ImageEnhance
import os, uuid

def augmentar_imagenes(lista_rutas, carpeta_resultados):
    resultados = []

    for ruta in lista_rutas:
        img = Image.open(ruta)
        base = os.path.splitext(os.path.basename(ruta))[0]

        # Lista de transformaciones
        transformaciones = {
            "rot90": img.rotate(90, expand=True),
            "rot180": img.rotate(180, expand=True),
            "rot270": img.rotate(270, expand=True),
            "flip_h": img.transpose(Image.FLIP_LEFT_RIGHT),
            "flip_v": img.transpose(Image.FLIP_TOP_BOTTOM),
            "brillo+": ImageEnhance.Brightness(img).enhance(1.5),
            "brillo-": ImageEnhance.Brightness(img).enhance(0.7),
            "zoom": img.resize((int(img.width * 1.2), int(img.height * 1.2)))
        }

        for nombre, im in transformaciones.items():
            out_name = f"{base}_{nombre}_{uuid.uuid4().hex[:6]}.png"
            out_path = os.path.join(carpeta_resultados, out_name)
            im.save(out_path)
            resultados.append({
                "nombre": out_name,
                "ruta": out_path.replace("\\", "/")
            })

    return resultados