import os
from PIL import Image

DATASET = "/home/kmonti/Desktop/dataset_preprocesado"

bad_files = []

for root, _, files in os.walk(DATASET):
    for f in files:
        if f.lower().endswith(".png"):
            path = os.path.join(root, f)
            try:
                img = Image.open(path)
                img.verify()
            except Exception as e:
                print("❌ PNG dañado:", path, "| Error:", e)
                bad_files.append(path)

print("\nEncontrados:", len(bad_files), "archivos dañados.")