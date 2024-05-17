from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define the class names
class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Tomato', 'Radish', 'Pumpkin', 'Bean']

# Load the pre-trained model
model_path = 'C:/Users/Saireena/Desktop/majorproject/vegetable_classifier_model.h5'
model = load_model(model_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        # Save the file to the 'static' folder
        upload_dir = os.path.join(os.getcwd(), 'static')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        img_path = os.path.join(upload_dir, file.filename)
        file.save(img_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(150, 150))  # Resize to (150, 150)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        return render_template('result.html', image_file=img_path, prediction=predicted_class)

@app.route('/logout', methods=['POST'])
def logout():
    if request.method == 'POST':
        # Redirect to logout page
        return render_template('logout.html')

if __name__ == '__main__':
    app.run(debug=True)
