from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model, model_from_json
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Cargamos el modelo entrenado
json_file = open('models/model_ft.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/model_ft.h5")
loaded_model._make_predict_function()          # Necesario!!
print("¡Modelo cargado de disco a memoria!. Verificar en http://127.0.0.1:5000/")


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocesando la imagen
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Página principal
    return render_template('login.html')

@app.route('/main', methods=['GET','POST'])
def mostrarMain():
    #Menú principal
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Obtiene el archivo de un post request
        f = request.files['file']

        # Guarda el archivo en la siguente ruta ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Realiza la predicción
        preds = model_predict(file_path, loaded_model)
        # print(type(preds))
        # print(preds.shape)
        # print("Primer valor: " + str(preds[0,0]) + ", segundo valor " + str(preds[0,1]))
        
        if preds[0,0] == 1.0:
            resultString = "Healthy"
            return resultString

        elif preds[0,1] == 1.0:
            resultString = "Retinopathy"
            return resultString

        else:
            resultString = "Error interno en el modelo"
            return resultString
        
    return None


if __name__ == '__main__':
    app.run(debug=True)

