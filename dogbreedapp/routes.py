from dogbreedapp import app
from flask import render_template, request, redirect
from flask import send_from_directory
from wrangling_scripts.CNN_functions import Resnet50_full_algorithm

import os

app.config["IMAGE_UPLOADS"] = "/home/dcalvomayo/Data/Capstone_dogs/app/dogbreedapp/images"

@app.route('/')
@app.route('/index')
def index():

    return render_template('index.html')

@app.route('/upload-file', methods=['POST'])
def upload_file():
    if request.files:
        image = request.files["image"]
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
        #print("Image saved")
        img_path = '/images/' + image.filename
        text = Resnet50_full_algorithm('dogbreedapp/images/' + image.filename)
        #print(text)
        return render_template('index.html', value=text, img_path=img_path)

@app.route('/images/<path:filename>')
def download_file(filename):
    return send_from_directory('images', filename, as_attachment=True)



