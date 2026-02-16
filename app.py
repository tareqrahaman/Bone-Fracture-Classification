import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
os.environ['TF_USE_LEGACY_KERAS'] = '1'

app = Flask(__name__)

# 1. Load your trained model
def load_fracture_model(model_path, num_classes):
    # 1. Rebuild the exact same architecture
    vgg_base = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = Sequential([
        vgg_base,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # 2. Load the weights from your saved .h5 file
    # Even if you used save_model, this will pull the weights out correctly
    model.load_weights(model_path)
    return model

# Usage (replace '2' with your actual number of classes)
MODEL_PATH = 'model/fracture_model.h5'
model = load_fracture_model(MODEL_PATH, num_classes=2)

# Define your classes (Ensure the order matches your training indices)
CLASS_NAMES = ['Oblique fracture', 'Spiral Fracture'] 

def model_predict(img_path, model):
    # 2. Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224)) # VGG16 standard size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) # Normalizes pixels based on VGG16 requirements

    # 3. Perform prediction
    preds = model.predict(x)
    # Get the index of the highest probability
    result_index = np.argmax(preds, axis=1)[0]

    # Calculate confidence percentage
    confidence = np.max(preds) * 100 
    
    return CLASS_NAMES[result_index], f"{confidence:.2f}%"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"
    
    f = request.files['file']
    if f.filename == '':
        return "No selected file"

    # Save the file temporarily
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'static/uploads', f.filename)
    f.save(file_path)

    # Run Prediction
    result, score = model_predict(file_path, model)

    return render_template('index.html', 
                           prediction=result, 
                           confidence=score, 
                           image_path=f.filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)