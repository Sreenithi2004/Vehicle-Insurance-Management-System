from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# Load your dataset (assuming it's a dictionary with image titles as keys and image data as values)
dataset = {
    'A': np.array(Image.open('A_image_path.jpg')),
    'B': np.array(Image.open('B_image_path.jpg')),
    # Add more entries for other letters/numbers
}

# Define a function to process uploaded images
def process_image(image):
    # Preprocess the image (resize, normalize, etc.)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

# Define a function to match uploaded image with dataset
def match_image(image):
    best_match = None
    max_similarity = -1
    for label, dataset_image in dataset.items():
        similarity = ssim(image, dataset_image, multichannel=True)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = label
    return best_match

# Define a route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image = Image.open(file)
        processed_image = process_image(image)
        # Match uploaded image with dataset
        result = match_image(processed_image)
        return jsonify({'result': result})
    return jsonify({'error': 'Error processing file'})

if __name__ == '__main__':
    app.run(debug=True)
