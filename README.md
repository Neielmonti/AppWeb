# Instrucciones de uso

## APLICACION FLASK

### Paso 1: Instale los requerimientos
Ejecute los siguientes comandos (en la carpeta raiz del proyecto):
- python3 -m venv venv
- source venv/bin/activate          # en Linux/Mac
- source venv\Scripts\activate      # en Windows
- pip install -r requirements.txt

### Paso 2: Corra el servidor
- python app.py

### Paso 3: Abra la direccion "http://127.0.0.1:5000" en su navegador
Hay diferentes direcciones para diferentes funcionalidades:
+ http://127.0.0.1:5000 -> Para analisis de imagenes (1 o varias)
+ http://127.0.0.1:5000/preprocesar -> Para preprocesar un dataset (subido en .zip)
+ http://127.0.0.1:5000/detectar_enfermedad -> Para detectar la enfermedad de una foto


## ENTRENAMIENTO DEL MODELO
Copie y pegue el archivo .env_example, y a la copia cambiele el nombre a ".env".
Luego, configure las variables del archivo.
Para iniciar el entrenamiento, ejecute el siguiente comando (en la carpeta raiz del proyecto):
- python model.py

Para poder utilizar el modelo recien entrenado, debe insertar los archivos .keras y .json en la carpeta {model},
con los siguientes nombres (eliminando los existentes):
+ modelo_soybean.keras
+ clases_soybean.json