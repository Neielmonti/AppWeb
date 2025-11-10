from flask import Flask, render_template, request, send_file
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
from analisis import analizar_imagen
from augmentation import augmentar_imagenes
import zipfile
import cv2
import uuid

UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "files" not in request.files:
            return "⚠️ No se enviaron archivos."

        files = request.files.getlist("files")
        resultados = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # Analizar imagen → devuelve estadísticas + ruta del histograma
                stats, hist_path = analizar_imagen(filepath, app.config["RESULTS_FOLDER"])

                resultados.append({
                    "filename": filename,
                    "filepath": filepath,
                    "stats": stats,
                    "histograma": hist_path
                })

        return render_template("resultados.html", resultados=resultados)

    return render_template("index.html")

@app.route("/augmentar", methods=["GET", "POST"])
def augmentar():
    if request.method == "POST":
        if "file" not in request.files:
            return "⚠️ No se envió ningún archivo ZIP."

        file = request.files["file"]
        if not file.filename.endswith(".zip"):
            return "⚠️ Solo se aceptan archivos ZIP."

        # Crear carpetas temporales
        temp_input = tempfile.mkdtemp()
        temp_output = os.path.join(app.config["RESULTS_FOLDER"], "augmented")
        os.makedirs(temp_output, exist_ok=True)

        # Guardar y descomprimir el ZIP
        zip_path = os.path.join(temp_input, "dataset.zip")
        file.save(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_input)

        # Ejecutar augmentación
        from augmentation import augment_dataset
        procesados = augment_dataset(temp_input, temp_output)

        # Crear ZIP de salida
        output_zip = os.path.join(app.config["RESULTS_FOLDER"], "dataset_augmented.zip")
        shutil.make_archive(output_zip.replace(".zip", ""), "zip", temp_output)

        return render_template("augment_resultados.html",
                               procesados=procesados,
                               zip_path="static/results/dataset_augmented.zip")

    return render_template("augment.html")

@app.route("/resize", methods=["GET", "POST"])
def resize():
    if request.method == "POST":
        if "file" not in request.files:
            return "⚠️ No se envió ningún archivo ZIP."

        file = request.files["file"]
        if not file.filename.endswith(".zip"):
            return "⚠️ Solo se aceptan archivos ZIP."

        temp_input = tempfile.mkdtemp()
        temp_output = os.path.join(app.config["RESULTS_FOLDER"], "resized")
        os.makedirs(temp_output, exist_ok=True)

        zip_path = os.path.join(temp_input, "dataset.zip")
        file.save(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_input)

        from augmentation import resize_dataset
        procesados = resize_dataset(temp_input, temp_output, scale=0.3)

        # Crear ZIP
        output_zip = os.path.join(app.config["RESULTS_FOLDER"], "dataset_resized.zip")
        shutil.make_archive(output_zip.replace(".zip", ""), "zip", temp_output)

        return render_template("resize_resultados.html",
                               procesados=procesados,
                               zip_path="static/results/dataset_resized.zip")

    return render_template("resize.html")

@app.route("/exagerar", methods=["GET", "POST"])
def exagerar():
    if request.method == "POST":
        if "file" not in request.files:
            return "⚠️ No se envió ningún archivo ZIP."

        file = request.files["file"]
        if not file.filename.endswith(".zip"):
            return "⚠️ Solo se aceptan archivos ZIP."

        temp_input = tempfile.mkdtemp()
        temp_output = os.path.join(app.config["RESULTS_FOLDER"], "exageradas")
        os.makedirs(temp_output, exist_ok=True)

        zip_path = os.path.join(temp_input, "dataset.zip")
        file.save(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_input)

        from exageracion_hsv import exagerar_dataset
        procesadas = exagerar_dataset(temp_input, temp_output)

        output_zip = os.path.join(app.config["RESULTS_FOLDER"], "dataset_exageradas.zip")
        shutil.make_archive(output_zip.replace(".zip", ""), "zip", temp_output)

        return render_template("exagerar_resultados.html",
                               procesadas=procesadas,
                               zip_path="static/results/dataset_exageradas.zip")

    return render_template("exagerar.html")

@app.route("/estandarizar", methods=["GET", "POST"])
def estandarizar():
    if request.method == "POST":
        if "file" not in request.files:
            return "⚠️ No se envió ningún archivo ZIP."

        file = request.files["file"]
        if not file.filename.endswith(".zip"):
            return "⚠️ Solo se aceptan archivos ZIP."

        temp_input = tempfile.mkdtemp()
        temp_output = os.path.join(app.config["RESULTS_FOLDER"], "estandarizadas")
        os.makedirs(temp_output, exist_ok=True)

        zip_path = os.path.join(temp_input, "dataset.zip")
        file.save(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_input)

        from estandarizar_tamano import estandarizar_dataset
        procesadas = estandarizar_dataset(temp_input, temp_output)

        output_zip = os.path.join(app.config["RESULTS_FOLDER"], "dataset_estandarizadas.zip")
        shutil.make_archive(output_zip.replace(".zip", ""), "zip", temp_output)

        return render_template("estandarizar_resultados.html",
                               procesadas=procesadas,
                               zip_path="static/results/dataset_estandarizadas.zip")

    return render_template("estandarizar.html")

@app.route("/detectar_enfermedad", methods=["GET", "POST"])
def detectar_enfermedad():
    import tensorflow as tf
    import numpy as np
    import json

    if request.method == "POST":
        if "file" not in request.files:
            return "⚠️ No se envió ningún archivo de imagen."

        file = request.files["file"]
        if file.filename == "":
            return "⚠️ No se seleccionó ningún archivo."

        from estandarizar_tamano import estandarizar_imagen
        from exageracion_hsv import exagerar_imagen, procesar_iluminacion

        # --- Guardar imagen subida ---
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

        # --- Leer imagen original ---
        img = cv2.imread(input_path)
        if img is None:
            return "❌ Error al leer la imagen. Asegúrate de subir un archivo válido."

        # --- Corregir iluminación ---
        img_corr = procesar_iluminacion(img)
        corrected_path = os.path.join(app.config["RESULTS_FOLDER"], f"corr_{filename}")
        cv2.imwrite(corrected_path, img_corr)

        # --- Estandarizar tamaño ---
        standardized_path = os.path.join(app.config["RESULTS_FOLDER"], f"std_{filename}")
        estandarizar_imagen(corrected_path, standardized_path)

        # --- Exagerar colores ---
        exaggerated_path = os.path.join(app.config["RESULTS_FOLDER"], f"exag_{filename}")
        exagerar_imagen(standardized_path, exaggerated_path)

        # =============================
        # 🧠 PREDICCIÓN CON EL MODELO
        # =============================
        MODEL_PATH = r"modelo_soybean.h5"
        CLASSES_JSON = r"clases_soybean.json"

        # Cargar modelo
        model = tf.keras.models.load_model(MODEL_PATH)

        # Cargar clases desde JSON
        try:
            with open(CLASSES_JSON, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        except Exception as e:
            return f"❌ Error al cargar clases: {e}"

        # Preprocesar imagen igual que en entrenamiento
        img = tf.keras.utils.load_img(exaggerated_path, target_size=(640,360))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, 0) / 255.0  # normalización

        # Hacer predicción
        predictions = model.predict(img_array)
        pred_idx = np.argmax(predictions[0])
        pred_class = class_names[pred_idx]
        pred_conf = float(np.max(predictions[0]) * 100)

        # --- Mostrar resultado ---
        return render_template(
            "detectar_resultado.html",
            filename=f"exag_{filename}",
            pred_class=pred_class,
            pred_conf=round(pred_conf, 2)
        )

    # GET: mostrar formulario
    return render_template("detectar_enfermedad.html")

if __name__ == "__main__":
    app.run(debug=True)