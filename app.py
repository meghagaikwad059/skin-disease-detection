from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
    MODEL_PATH = os.path.join(MODEL_DIR, 'skin_disease_model.h5')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

app.config.from_object(Config)

# Global model variable
model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def load_model():
    global model
    try:
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)
            
        model = tf.keras.models.load_model('C:/Users/gaikw/OneDrive/Desktop/SkinDiseaseDetection/src/backend/skin_disease_model.h5')
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

def save_model():
    try:
        model.save(Config.MODEL_PATH)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def preprocess_image(image):
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        
        # Validate image dimensions
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Invalid image shape after processing: {img_array.shape}")
            
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded', 'status': 503}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided', 'status': 400}), 400
        
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 400}), 400
        
    if not allowed_file(image_file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed types: {", ".join(Config.ALLOWED_EXTENSIONS)}',
            'status': 400
        }), 400
    
    try:
        # Secure filename and read image
        filename = secure_filename(image_file.filename)
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Process and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        classes = ['melanoma', 'nevus', 'keratosis']
        confidence = float(np.max(prediction))
        
        result = {
            'disease': classes[np.argmax(prediction)],
            'confidence': confidence,
            'status': 200,
            'filename': filename
        }
        
        # Add warning if confidence is low
        if confidence < 0.7:
            result['warning'] = 'Low confidence prediction'
        
        return jsonify(result)
    
    except (IOError, ValueError) as e:
        return jsonify({'error': f'Invalid image: {str(e)}', 'status': 400}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}', 'status': 500}), 500

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'model_loaded': model is not None,
        'status': 'ready' if model else 'model not loaded',
        'api_version': '1.0'
    }
    
    if model:
        status.update({
            'model_input_shape': model.input_shape,
            'model_summary': str(model.summary())
        })
    
    return jsonify(status)

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)