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
REPORTE_METRICAS = r"model/reporte.csv"
CONFUSION_MATRIX_CSV = r"model/confusion_matrix.csv"

# MODELO BINARIO (para diferenciar entre SOJA y NaO_SOJA)
BINARIO_MODEL_PATH = r"binary_model/modelo_binario.keras"
BINARIO_CLASSES_JSON = r"binary_model/clases_binario.json"

modelo_binario = tf.keras.models.load_model(BINARIO_MODEL_PATH)
with open(BINARIO_CLASSES_JSON, "r") as f:
    clases_binarias = json.load(f)

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
    # ================================================
    # 1) GRÁFICO F1-SCORE POR CLASE
    # ================================================
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

        fig_f1.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside',
            marker_color="rgba(66,188,113,0.8)"
        )

        fig_f1.update_layout(
            yaxis_range=[0, 1.1],
            margin=dict(l=30, r=30, t=10, b=0),
            paper_bgcolor="rgb(6,50,40)",
            plot_bgcolor="rgb(6,50,40)",
            font=dict(color="white")
        )

        fig_f1.update_xaxes(
            color="white",
            tickfont=dict(size=12)
        )
        fig_f1.update_yaxes(
            color="white",
            tickfont=dict(size=12)
        )

    except Exception as e:
        print(f"❌ Error cargando gráfico F1: {e}")
        fig_f1 = None

    # ================================================
    # 2) MATRIZ DE CONFUSIÓN
    # ================================================
    try:
        df_cm = pd.read_csv(CONFUSION_MATRIX_CSV, index_col=0)

        # Definir la escala de color personalizada
        # 1. Color de inicio (oscuro, cercano al fondo)
        start_color = "rgb(15, 65, 50)" 
        # 2. Color de fin (tu verde deseado para los valores altos)
        end_color = "rgb(66, 188, 113)" 
        
        # Crear una escala secuencial simple de dos tonos
        custom_scale = [
            [0.0, start_color],
            [1.0, end_color]
        ]

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
            # Usar la nueva escala personalizada
            color_continuous_scale=custom_scale
        )

        # Ejes y texto
        fig_cm.update_xaxes(
            side="top",
            tickangle=45,
            tickfont=dict(size=12, color="white"),
            title_font=dict(size=14, color="white")
        )
        fig_cm.update_yaxes(
            autorange="reversed",
            tickfont=dict(size=12, color="white"),
            title_font=dict(size=14, color="white")
        )

        # Mantener el texto blanco, ahora será legible sobre el verde (66, 188, 113)
        fig_cm.update_traces(
            textfont=dict(size=14, color="white")
        )

        fig_cm.update_layout(
            height=400,
            width=None,
            margin=dict(l=200, r=200, t=0, b=10),
            paper_bgcolor="rgb(6,50,40)",
            plot_bgcolor="rgb(6,50,40)",
            font=dict(color="white")
        )

    except Exception as e:
        print(f"❌ Error cargando matriz de confusión: {e}")
        fig_cm = None

    # ================================================
    # 3) CREAR APP DASH CON ESTILO GLOBAL
    # ================================================
    dash_app = Dash(
        __name__,
        server=app,
        url_base_pathname="/dash/"
    )

    # Estilo global para h2 y h3
    dash_app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Métricas del Modelo</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    background-color: rgb(6,50,40);
                }
                h2 {
                    text-align: center;
                    color: white !important;
                    font-family: "Montserrat", sans-serif !important;
                }
                h3 {
                    color: white !important;
                    font-family: Arial, sans-serif !important;
                    text-align: left;
                    width: 80%;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """

    # ================================================
    # 4) LAYOUT DASHBOARD
    # ================================================
    layout_items = [ html.H2("") ]

    if fig_f1:
        layout_items.append(html.H3("F1-Score por Clase"))
        layout_items.append(dcc.Graph(figure=fig_f1,style={"width": "75%", "height": "15rem"}))
        
        # 💡 AÑADIR SEPARADOR BLANCO AQUÍ
        layout_items.append(
            html.Div(style={
                "borderTop": "2px solid white",  # Línea blanca sólida de 2px
                "margin": "10px 10px",             # Margen superior e inferior para espacio
                "width": "90%",                 # Ancho de la línea
                "marginLeft": "auto",           # Centrar
                "marginRight": "auto"           # Centrar
            })
        )
        

    if fig_cm:
        layout_items.append(html.H3("Matriz de Confusión"))
        layout_items.append(dcc.Graph(figure=fig_cm,style={"width": "70%", "height": "70%"}))

    # ================================================
    # 4) LAYOUT DASHBOARD - APLICAR ESTILOS FLEXBOX AQUÍ
    # ================================================
    dash_app.layout = html.Div(
        layout_items,
        # Agregamos el estilo al contenedor principal (html.Div)
        style={
            "display": "flex",          # Activa Flexbox
            "flexDirection": "column",  # Apila los elementos verticalmente (H2, H3, Graph)
            "alignItems": "center"      # Centra los elementos a lo largo del eje transversal (horizontalmente en este caso)
        }
    )


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

nombres_legibles = {
    "Septoria_Glycines": "Septoria Glycines",
    "bacterial_blight": "Bacterial Blight",
    "cercospora_leaf_blight": "Cercospora Leaf Blight",
    "downey_mildew": "Downey Mildew",
    "frogeye": "Frogeye",
    "healthy": "Saludable",
    "potassium_deficiency": "Potassium Deficiency",
    "soybean_rust": "Soybean Rust",
    "target_spot": "Target Spot"
}

@app.route("/detectar_enfermedad", methods=["GET", "POST"])
def detectar_enfermedad():
    if request.method == "POST":

        # -------------------------
        # 1: VALIDAR Y GUARDAR IMAGEN RAW
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
        # 2: CHECKEAR SI ES SOJA O NO_SOJA
        # -------------------------
        img = tf.keras.utils.load_img(raw_path, target_size=(224, 224))
        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, 0)

        pred_bin = modelo_binario.predict(arr)[0][0]

        if pred_bin < 0.5:
            return render_template(
                "detectar_resultado.html",
                filename="no_soja.jpg",
                pred_class="❌ La imagen no parece ser soja.",
                pred_conf=round(float(1 - pred_bin) * 100, 2)
            )

        # -------------------------
        # 3: APLICAR PIPELINE COMPLETO
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
        # 4: VERIFICAR MODELO GLOBAL
        # -------------------------
        if model is None:
            return "❌ Error fatal: El modelo no está cargado en el servidor."

        # -------------------------
        # 5: PREPARAR IMAGEN PARA EL MODELO
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
        # 6: PREDICCIÓN
        # -------------------------
        try:
            # El modelo recibe [0, 255] y él mismo lo procesa por dentro
            predictions = model.predict(img_array) 
        except Exception as e:
            return f"❌ Error durante la predicción: {e}"

        # -------------------------
        # 7: MOSTRAR RESULTADO (MENSAJES SEMÁNTICOS)
        # -------------------------
        pred_idx = int(np.argmax(predictions[0]))
        raw_class = class_names[pred_idx]
        
        # Obtenemos el nombre amigable
        nombre_display = nombres_legibles.get(raw_class, raw_class.replace("_", " ").title())
        pred_conf = float(np.max(predictions[0]) * 100)

        # Configuración de umbrales
        X = 20  # Confianza mínima
        Y = 70  # Confianza alta

        if pred_conf < X:
            mensaje = "⚠️ El sistema no está seguro de qué le ocurre a la hoja."
        
        elif raw_class == "healthy":
            # Caso especial para plantas sanas
            if pred_conf < Y:
                mensaje = f"🤔 La planta parece estar mayormente {nombre_display.lower()}."
            else:
                mensaje = f"✅ La planta se encuentra {nombre_display.lower()}."
        
        elif raw_class == "potassium_deficiency":
            # Caso especial para deficiencias (no es una enfermedad infecciosa)
            if pred_conf < Y:
                mensaje = f"🤔 Podría haber una {nombre_display.lower()}."
            else:
                mensaje = f"⚠️ Se ha identificado {nombre_display.lower()}."
        
        else:
            # Caso general para enfermedades
            if pred_conf < Y:
                mensaje = f"🤔 Posible presencia de {nombre_display}."
            else:
                mensaje = f"❌ Se ha detectado {nombre_display} con alta certeza."

        return render_template(
            "detectar_resultado.html",
            filename=filename,
            file_folder="uploads",
            pred_class=mensaje,
            pred_conf=round(pred_conf, 2)
        )

    # GET → mostrar formulario
    return render_template("detectar_enfermedad.html")

if __name__ == "__main__":
    app.run(debug=True)