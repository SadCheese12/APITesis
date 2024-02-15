from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pickle
import os
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import tensorflow as tf
from joblib import load
import cv2
from collections.abc import MutableMapping
from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://monterosuniga:monterosunigadb@cluster0.6sh1gzt.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)
db = client["BaseScrapping"]

with open("modelos/clustering/kmeans_model.pkl", "rb") as file:
    kmeans_model = load(file)
# Carga el Autoencoder de 16 colores
autoencoder = tf.keras.models.load_model("modelos/autoencoder/modelo256CV2-0.tf")

#Cargo el modelo para reducir la imagen a 16 colores
with open("modelos/color/kmeans_model256colorsReduction.pkl", "rb") as file:
    color_reduction16 = load(file)

with open("modelos/umap/umap_model.pkl", "rb") as file:
    umap_model = load(file)

app = Flask(__name__)

database_elements = [
    {"id": 1, "nombre": "Elemento 1", "descripcion": "Descripción 1"},
    {"id": 2, "nombre": "Elemento 2", "descripcion": "Descripción 2"},
    # ... otros elementos de la base de datos
    
]

@app.route("/get_cluster/<cluster>", methods=["GET"])
def get_data_from_mongodb(cluster):
    try:
        resultados = db.Imgs_AE_UMAP_Model_Kmeans.find({"id_cluster": cluster}, {"_id": 0})
        resultados_list = []
        for documento in resultados:
            documento["_id"] = str(documento.get("_id"))
            resultados_list.append(documento)
        return jsonify(resultados_list)
    except Exception as e:
        return jsonify({"error": str(e)})



@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    if file:
        filename = secure_filename("imgPOST.png")
        filepath = os.path.join("imgPOST", filename)
        file.save(filepath)
        cluster = process_image(filepath)
        return jsonify({"cluster": cluster})

def process_image(filepath):
    image = cv2.imread(filepath)#Se lee la imagen
    image = cv2.resize(image, (240, 240))#Se redimensiona 
    image = image.astype("float32") / 255.0#Se normaliza
    image = image.reshape(1*240*240,3)#Se redimensiona a 2D pa reducir los colores
    image = color_reduction16.cluster_centers_[color_reduction16.predict(image)]#Se reduce a 256 colores
    image = image.reshape(1,240,240,3)#Se redimensiona a 4D para el encoder
    image = autoencoder.encoder(image).numpy()#Procesa el encoder
    image = umap_model.transform(image)#Se transforma con umap
    image = kmeans_model.predict(image)#Se predice el cluster
    cluster = int(image[0])
    return cluster#Se retorna el cluster

if __name__ == "__main__":
    app.run(debug=True)
