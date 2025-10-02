from flask import Flask, render_template, request, send_file
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
from analisis import analizar_imagen
from augmentation import augmentar_imagenes
import zipfile
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
        if "files" not in request.files:
            return "⚠️ No se enviaron archivos."

        files = request.files.getlist("files")
        imagenes = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                imagenes.append(filepath)

        # Generar dataset aumentado → devuelve lista [(nombre, ruta)] o zip
        resultados = augmentar_imagenes(imagenes, app.config["RESULTS_FOLDER"])

        # 🔹 Si querés devolver un ZIP para descargar
        # return send_file(resultados, as_attachment=True, download_name="dataset_aumentado.zip")

        # 🔹 Si preferís mostrar una galería en la web:
        return render_template("augment_resultados.html", resultados=resultados)

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

if __name__ == "__main__":
    app.run(debug=True)