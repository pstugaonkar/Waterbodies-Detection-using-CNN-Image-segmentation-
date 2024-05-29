from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('model.h5')

# Define a function to prepare the image
def prepare_image(img_path, img_height=128, img_width=128):
    # Read and resize the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalizing the image array
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Please provide a file."
    
    file = request.files['file']
    if file.filename == '':
        return "Please provide a file."
    
    # Save the file to the server
    filepath = r"C:\Users\pstug\Desktop\ocean\uploads\water_body_17.jpg"
    file.save(filepath)
    
    # Prepare the image
    img_array = prepare_image(filepath)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_mask = prediction.reshape((128, 128))  # Reshape the prediction to match the mask shape
    
    # Clean up: Remove the uploaded file after prediction
    os.remove(filepath)
    
    # Calculate the water body percentage
    image_mean = predicted_mask.mean()
    fraction = image_mean
    percentage = fraction * 100
    formatted_percentage = '{:.2f}%'.format(round(percentage, 2))
    sq = 21086.64
    waterbodies = float(image_mean) * sq
    
    return f'Predicted water body coverage: {formatted_percentage}, Approximate area: {waterbodies:.2f} square feet'

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
