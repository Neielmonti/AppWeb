# Instrucciones de uso

### Paso 1: Instale los requerimientos
- python3 -m venv venv
- source venv/bin/activate   # en Linux/Mac
- venv\Scripts\activate      # en Windows
- pip install -r requirements.txt

### Paso 2: Corra el servidor
- python app.py

### Paso 3: Abra la direccion "http://127.0.0.1:5000" en su navegador
o "http://127.0.0.1:5000/augmentar" para aumentar el dataset.

### Paso 4: App Web
- http://127.0.0.1:5000 -> Para analisis de imagenes (1 o varias)
- http://127.0.0.1:5000/resize -> Para reducir el tamaño de las imagenes a un 33% (respeta estructura de directorios)
- http://127.0.0.1:5000/augmentar -> Para augmentar el dataset (respeta estructura de directorios)
