# debug_predict.py
import os, json, numpy as np, tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

MODEL_PATH = "modelo_soybean.keras"  # <- OJO: poné el path real que generó entrenamiento
CLASSES_JSON = "clases_soybean.json"
TEST_IMG = "/home/kmonti/Desktop/dataset_preprocesado/healthy/healthy_513_proc_zoom2_4b470f.png"  # elegí una imagen del dataset

print("model exists:", os.path.exists(MODEL_PATH))
print("json exists:", os.path.exists(CLASSES_JSON))
with open(CLASSES_JSON,'r') as f:
    classes = json.load(f)
print("classes len:", len(classes), "first classes:", classes[:5])

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded. summary:")
model.summary()

# Load image as you do in the endpoint:
img = image.load_img(TEST_IMG, target_size=(224,224))
arr = image.img_to_array(img)
arr = np.expand_dims(arr,0)
arr = preprocess_input(arr)
print("input stats: min,mean,max:", arr.min(), arr.mean(), arr.max(), "shape:", arr.shape)

preds = model.predict(arr, verbose=0)[0]
print("top probs:", sorted(list(enumerate(preds)), key=lambda x:-x[1])[:5])
print("argmax:", np.argmax(preds), "class:", classes[np.argmax(preds)], "conf:", preds[np.argmax(preds)])

print("Weights loaded:", "conv1" in [w.name for w in model.layers[4].weights])