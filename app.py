import json
import os
import shutil
import tempfile
import uuid
import zipfile
import cv2
import numpy as np
import tensorflow as tf
from flask import (
    Flask, 
    render_template, 
    request, 
    redirect, 
    send_file,
    send_from_directory
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename
from analisis import analizar_imagen
from augmentation import augmentar_imagenes
from pipeline import procesar_imagen_pipeline
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# =========================================================
# CARGA GLOBAL
# =========================================================
print("Cargando modelo de Keras y clases...")

MODEL_PATH = r"model/modelo_soybean.keras"
CLASSES_JSON = r"model/clases_soybean.json"
REPORTE_METRICAS = r"model/model_report.csv"
CONFUSION_MATRIX_CSV = r"model/confusion_matrix.csv"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_JSON, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    print("✅ Modelo y clases cargados exitosamente.")
except Exception as e:
    print(f"❌ ERROR FATAL: No se pudo cargar el modelo o las clases: {e}")
    # En un escenario real, quizás quieras que la app falle si no puede cargar
    model = None
    class_names = []


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def create_dash_app(app):
    # =========================================================
    # GRÁFICO 1: F1-SCORE POR CLASE (ORIGINAL)
    # =========================================================
    try:
        df_f1 = pd.read_csv(REPORTE_METRICAS)
        df_f1 = df_f1.rename(columns={
            "Unnamed: 0": "clase",
            "f1-score": "f1"
        })
        fig_f1 = px.bar(
            df_f1,
            x="clase",
            y="f1",
            text="f1",
            title=""
        )
        fig_f1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_f1.update_layout(yaxis_range=[0, 1])
    except Exception as e:
        print(f"❌ Error al cargar o generar el gráfico F1-Score: {e}")
        fig_f1 = None


    # =========================================================
    # GRÁFICO 2: MATRIZ DE CONFUSIÓN (AJUSTE FINAL DE MÁRGENES)
    # =========================================================
    try:
        df_cm = pd.read_csv(CONFUSION_MATRIX_CSV, index_col=0)

        fig_cm = px.imshow(
            df_cm.values, 
            x=df_cm.columns.tolist(), 
            y=df_cm.index.tolist(),
            text_auto=True,
            labels={
                "x": "Clase Predicha",
                "y": "Clase Verdadera",
                "color": "Conteo"
            },
            title="",
            color_continuous_scale=px.colors.sequential.Viridis 
        )

        # Mejoras visuales:
        fig_cm.update_xaxes(
            side="top", 
            tickangle=45, 
            title_font_size=14
        ) 
        fig_cm.update_yaxes(
            title_font_size=14,
            autorange="reversed" 
        )
        
        # Ajustar el tamaño del texto dentro de las celdas
        fig_cm.update_traces(
            textfont=dict(size=10, color='white') 
        )

        # AJUSTES DE MÁRGENES CLAVE PARA EVITAR CHOQUES (t=superior, b=inferior)
        fig_cm.update_layout(
            title_font_size=20,
            height=800, 
            width=800,
            # Se aumentó 't' (top) a 80 y 'b' (bottom) a 180 para más espacio
            margin=dict(l=280, r=10, t=180, b=180), 
            coloraxis_colorbar=dict(title="Conteo")
        )
        
    except Exception as e:
        print(f"❌ Error al cargar o generar el gráfico de Matriz de Confusión: {e}")
        fig_cm = None


    dash_app = Dash(
        __name__,
        server=app,
        url_base_pathname="/dash/"
    )

    # =========================================================
    # LAYOUT DEL DASHBOARD CON AMBOS GRÁFICOS
    # =========================================================
    layout_items = [
        html.H2("Métricas del Modelo - Soja"),
    ]

    if fig_f1:
        layout_items.append(html.H3("F1-Score por Clase"))
        layout_items.append(dcc.Graph(figure=fig_f1))

    if fig_cm:
        layout_items.append(html.H3("Matriz de Confusión"))
        layout_items.append(dcc.Graph(figure=fig_cm))

    dash_app.layout = html.Div(layout_items)


# Inicializar Dash
create_dash_app(app)

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

@app.route("/preprocesar", methods=["GET", "POST"])
def preprocesar():
    if request.method == "POST":
        if "file" not in request.files:
            return "⚠️ No se envió ningún archivo ZIP."

        file = request.files["file"]
        if not file.filename.endswith(".zip"):
            return "⚠️ Solo se aceptan archivos ZIP."

        import zipfile
        import tempfile
        import shutil
        from pipeline import procesar_imagen_pipeline
        from augmentation import augment_dataset

        # carpetas temporales
        temp_input = tempfile.mkdtemp()
        temp_preproc = tempfile.mkdtemp()
        temp_aug = tempfile.mkdtemp()

        # guardar zip
        zip_path = os.path.join(temp_input, "dataset.zip")
        file.save(zip_path)

        # extraer
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_input)

        # === 1) preprocesamiento ===
        for root, _, files in os.walk(temp_input):
            for f in files:
                if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    continue

                ruta_abs = os.path.join(root, f)
                rel = os.path.relpath(ruta_abs, temp_input)

                out_path = os.path.join(temp_preproc, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                out_path = out_path.rsplit(".", 1)[0] + "_proc.png"
                procesar_imagen_pipeline(ruta_abs, out_path)

        # === 2) augmentación (mantiene carpetas) ===
        augment_dataset(temp_preproc, temp_aug)

        # === 3) crear ZIP final ===
        output_base = os.path.join(app.config["RESULTS_FOLDER"], "dataset_preprocesado")
        shutil.make_archive(output_base, "zip", temp_aug)
        output_zip = output_base + ".zip"

        # === 4) mostrar template de resultado ===
        return render_template(
            "preprocesar_resultados.html",
            zip_filename=os.path.basename(output_zip)
        )

    return render_template("preprocesar.html")

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(
        app.config["RESULTS_FOLDER"],
        filename,
        as_attachment=True,
        mimetype="application/zip"
    )

@app.route("/detectar_enfermedad", methods=["GET", "POST"])
def detectar_enfermedad():
    if request.method == "POST":

        # -------------------------
        # 1: VALIDAR Y GUARDAR IMAGEN RAW (¡ESTE CÓDIGO FALTABA!)
        # -------------------------
        if "file" not in request.files:
            return "⚠️ No se envió ningún archivo de imagen."

        file = request.files["file"]
        if file.filename == "":
            return "⚠️ No se seleccionó ningún archivo."

        if not (file and allowed_file(file.filename)):
            return "⚠️ Tipo de archivo no permitido."

        # Guardar la imagen original
        filename = secure_filename(file.filename)
        raw_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(raw_path)

        # -------------------------
        # 2: APLICAR PIPELINE COMPLETO (¡ESTE CÓDIGO FALTABA!)
        # -------------------------
        # Aquí es donde se definen las variables 'proc_name' y 'proc_path'
        proc_name = f"{uuid.uuid4().hex}_proc.png"
        proc_path = os.path.join(app.config["RESULTS_FOLDER"], proc_name)

        try:
            # Llama a tu pipeline de CV (segmentación, etc.)
            ok = procesar_imagen_pipeline(raw_path, proc_path)
            if not ok:
                return "❌ Error procesando la imagen con el pipeline."
        except Exception as e:
            return f"❌ Error en pipeline de imagen: {e}"


        # -------------------------
        # 3: VERIFICAR MODELO GLOBAL
        # -------------------------
        if model is None:
            return "❌ Error fatal: El modelo no está cargado en el servidor."

        # -------------------------
        # 4: PREPARAR IMAGEN PARA EL MODELO
        # -------------------------
        IMG_HEIGHT = 224
        IMG_WIDTH = 224
        try:
            # ¡Ahora 'proc_path' SÍ existe y se puede usar aquí!
            img = tf.keras.utils.load_img(proc_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # La línea de 'preprocess_input' está correctamente eliminada
            # (tal como lo hiciste en tu último código)
            
        except Exception as e:
            return f"❌ Error preparando la imagen para el modelo: {e}"

        # -------------------------
        # 5: PREDICCIÓN
        # -------------------------
        try:
            # El modelo recibe [0, 255] y él mismo lo procesa por dentro
            predictions = model.predict(img_array) 
        except Exception as e:
            return f"❌ Error durante la predicción: {e}"

        # -------------------------
        # 6: MOSTRAR RESULTADO
        # -------------------------
        pred_idx = int(np.argmax(predictions[0])) 
        pred_class = class_names[pred_idx]
        pred_conf = float(np.max(predictions[0]) * 100)

        return render_template(
            "detectar_resultado.html",
            filename=proc_name, # ¡'proc_name' también existe ahora!
            pred_class=pred_class,
            pred_conf=round(pred_conf, 2)
        )

    # GET → mostrar formulario
    return render_template("detectar_enfermedad.html")

if __name__ == "__main__":
    app.run(debug=True)