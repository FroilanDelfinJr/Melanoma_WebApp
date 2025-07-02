# Melanoma Classification with U-Net and XGBoost
This project presents a machine learning application designed to classify skin lesions as either benign (NotMelanoma) or malignant (Melanoma). The system operates using a three-step process. The first step uses a U-Net model to identify and outline the lesion from an image. In the second step, a set of features are calculated based on the ABCD (Asymmetry, Border, Color, Diameter) rule. Finally, these features are analyzed by an XGBoost model to produce the final classification.

# Overview
The core of this project is a two-stage pipeline designed to mimic the diagnostic process of a dermatologist:

<ol>
  <li><b>Segmentation:</b> A U-Net deep learning model first analyzes the input image. Its purpose is to identify and outline the exact area of the skin lesion, separating it from the normal skin. This step produces a digital "mask" of the lesion's shape.</li>
  <li><b>Feature Extraction:</b> After the lesion is identified, the system calculates a set of key measurements based on the clinical ABCD rule for melanoma detection. These features include:
    <ul>
      <li><b>Asymmetry:</b> How uneven the shape of the lesion is.</li>
      <li><b>Border:</b> How irregular or poorly defined the lesion's edge is.</li>
      <li><b>Color:</b> The different colors and shades present across the lesion.</li>
      <li><b>Diameter:</b> The overall size of the lesion.</li>
      <li>Additional features related to shape and texture are also calculated.</li>
    </ul>
  </li>
  <li><b>Classification:</b> The extracted features are then passed to a trained XGBoost model. This model analyzes the measurements and makes a final prediction, classifying the lesion as either Melanoma or NotMelanoma.</li>
</ol>

# Technology Stack
<ul>
  <li><b>Backend:</b> Python, Flask</li>
  <li><b>Deep Learning (Segmentation):</b> TensorFlow, Keras</li>
  <li><b>Machine Learning (Classification):</b> Scikit-learn, XGBoost</li>
  <li><b>Image Processing:</b> OpenCV, Scikit-image</li>
  <li><b>Frontend:</b> HTML, CSS, JavaScript</li>
  <li><b>Data Handling:</b> Pandas, NumPy</li>
</ul>

# Project Structure
For the application to work, your project must follow this exact folder structure:

```
/Melanoma_WebApp/
│
├── models/
│   ├── Unet_model_final.h5
│   ├── xgb_model_final.json
│   └── pipeline_components.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
│
└── requirements.txt

```

# Setup and Installation
Follow these steps to run the application on your local machine.

Prerequisites
<ol>
  <li><b>Prerequisites</b>
    <ul>
      <li>Python 3.9+</li>
      <li>pip package manager</li>
    </ul>
  </li>
  <br>
  <li><b>Clone the Repository</b></li>
  Clone this repository to your local machine. This will include all necessary code and model files.<br>
  <pre lang=lisp>
  git clone https://github.com/FroilanDelfinJr/Melanoma_WebApp.git
  cd Melanoma_WebApp
  </pre>

  <li><b>Install Dependencies</b></li>
  This project uses a requirements.txt file to manage all necessary Python libraries. This is the most reliable way to set up the environment.
  <br>
  <br>
  In your terminal, run the following command from the main Melanoma_WebApp folder:
  <pre lang=lisp>
  pip install -r requirements.txt
  </pre>
  This will install the correct, compatible versions of Flask, TensorFlow, XGBoost, and all other required packages.
</ol>

# Running the Application
Once the setup is complete, you can start the Flask web server.

<ol>
  <li>Make sure you are in the main Melanoma_WebApp directory in your terminal.</li>
  <li>Run the following command:</li>
  <pre lang=lisp>
  python app.py
  </pre>
  <li>The terminal will show output indicating that the server is running, including a line like: * Running on http://127.0.0.1:5000</li>
  <li>Open your web browser and navigate to that address: http://127.0.0.1:5000</li>
</ol>
You should now see the web application's user interface, ready to accept image uploads.

# Acknowledgements
This project was trained on data from the following public datasets. A huge thank you to the organizers and contributors.

<ul>
  <li><b>ISIC (International Skin Imaging Collaboration) Archive:</b> Specifically, the ISIC 2018 datasets.</li>
  <li><b>HAM10000 ("Human Against Machine with 10000 Training Images") dataset.</b></li>
</ul>
