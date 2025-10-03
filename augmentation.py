from PIL import Image
import os, uuid

def zoom_image(img, factor):
    """
    Aplica un zoom manteniendo el mismo tamaño de salida.
    factor > 1 significa hacer zoom (ej: 1.2 -> 20% más cerca).
    """
    # Escalar la imagen
    new_size = (int(img.width * factor), int(img.height * factor))
    zoomed = img.resize(new_size, Image.LANCZOS)

    # Recortar al tamaño original, centrado
    left = (zoomed.width - img.width) // 2
    top = (zoomed.height - img.height) // 2
    right = left + img.width
    bottom = top + img.height

    return zoomed.crop((left, top, right, bottom))

def augmentar_imagenes(lista_rutas, carpeta_resultados):
    resultados = []

    for ruta in lista_rutas:
        img = Image.open(ruta)
        base = os.path.splitext(os.path.basename(ruta))[0]

        # 5 transformaciones: 2 flips + 3 zooms
        transformaciones = {
            "flip_h": img.transpose(Image.FLIP_LEFT_RIGHT),
            "flip_v": img.transpose(Image.FLIP_TOP_BOTTOM),
            "zoom1": zoom_image(img, 1.1),  # zoom leve
            "zoom2": zoom_image(img, 1.3),  # zoom medio
            "zoom3": zoom_image(img, 1.5)   # zoom fuerte
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

def resize_image(img, scale=0.3):
    """Reduce la imagen al porcentaje dado."""
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)

def augment_dataset(input_dir, output_dir):
    """
    Recorre un dataset (con subdirectorios), aplica augmentations y guarda
    las imágenes resultantes en la misma estructura de subdirectorios.
    """
    procesados = []
    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        out_dir = os.path.join(output_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        for file in files:
            in_path = os.path.join(root, file)
            try:
                with Image.open(in_path) as img:
                    base = os.path.splitext(file)[0]

                    # 5 augmentations
                    transformaciones = {
                        "flip_h": img.transpose(Image.FLIP_LEFT_RIGHT),
                        "flip_v": img.transpose(Image.FLIP_TOP_BOTTOM),
                        "zoom1": zoom_image(img, 1.1),
                        "zoom2": zoom_image(img, 1.3),
                        "zoom3": zoom_image(img, 1.5)
                    }

                    for nombre, im in transformaciones.items():
                        out_name = f"{base}_{nombre}_{uuid.uuid4().hex[:6]}.png"
                        out_path = os.path.join(out_dir, out_name)
                        im.save(out_path)
                        procesados.append(out_path)

                    print(f"Augmentada: {in_path} -> {out_dir}")

            except Exception as e:
                print(f"Error con {in_path}: {e}")

    return procesados