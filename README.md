# Dog breed classifier using Convolutional Neural Networks (CNNs).

### Introduction
This project is the last of [Udacity](www.udacity.com) Data Science Nanodegree required assignments.

The goal is to build a dog breed classifier based on CNNs, and deploy it to a web-based app. The classifier has been built step by step in a Jupyter notebook within one of Udacity workspaces. However, since not all the data used on it was accessible to be downloaded, only a html file has been included. You are more than encouraged to check it out!

In the end an accuracy of 80% is obtained in test data.

### Instructions
Apart from usual libraries, the following packages are required:
- tensorflow.keras which handles neural networks structures.
- flask to handle the back-end of the app.
- opencv-python to build the face detector.

After installing required libraries, all is required is to run the following command:

`python run.py`

and open a browser with the following value on the url:

`http://0.0.0.0:3001/`

### Files description:
**bottleneck_features**: Folder.
  - **DogResnet50Data.npz**: Bottleneck features from Resnet50 CNN.
  - **extract_bottleneck_features.py**: Python script containing functions required to extract bottleneck features.
**data**: Folder.
  - **dog_names.csv**: Csv file containing the different dog breeds.
**dog_app.html**: Jupyter notebook in html format containing all the steps to develop the classifier.
**dog_breed_app**: Folder.
  - **images**: Folder containing the images uploaded to the app.
  - **__init__.py: Python script that initializes Flask.
  - **routes.py**: Python script containing the back-end code.
  - **static/img**: Folder containing the images used by default in the app. 
  - **templates**: Folder.
    - **index.html**: Html code containing the front-end of the app.
**face_detector**: Folder containing the xml file used by the face detector.
**images**: Folder containing pictures that can be used to test the app. In particular, the ones inside validation_imgs correpond to the images used in Part 6 of the notebook to validate the algorithm. Please note that some of the images in images folder are used by the Jupyter notebook.
**run.py**: Python script to start the app.
**saved_models**: Folder containing the weighted factors used by the CNN.
**wrangling_scripts**: Folder.
  - **CNN_functions.py**: Python script containing all the functions required to run the CNN.

### How to interact?
For any questions you can contact me on dcalvomayo@gmail.com

Find this, and other cool projects on my [Github](https://github.com/dcalvomayo)

### Licensing
You may take any code, but don't forget to cite the source. Take into account that some code was developed by [Udacity](www.udacity.com).

### References:

Images obtained from the following sources:

https://www.lavanguardia.com/mascotas/20220326/8154265/son-razas-perro-mas-bonitas-mundo-nbs.html#foto-1

https://es.wikipedia.org/wiki/Familia_Airbus_A320#/media/Archivo:Lufthansa_Airbus_A320-211_D-AIQT_01.jpgV

https://www.eluniverso.com/noticias/internacional/corea-del-norte-califica-como-viejo-senil-a-joe-biden-nota/

https://ca.wikipedia.org/wiki/Ronaldo_de_Assis_Moreira

https://twitter.com/sanchezcastejon

https://es.wikipedia.org/wiki/Familia_Airbus_A320#/media/Archivo:Lufthansa_Airbus_A320-211_D-AIQT_01.jpg

https://unsplash.com/photos/eWqOgJ-lfiI
