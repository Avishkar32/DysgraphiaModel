import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'dysgraphia_model.h5'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    logger.info("\n=== Model Summary ===")
    model.summary(print_fn=lambda x: logger.info(x))
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(256, 256)):
    """
    Preprocess the image to match model requirements
    Returns: numpy array of shape (1, 256, 256, 3)
    """
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        logger.info(f"Preprocessed image shape: {img_array.shape}")
        logger.info(f"Pixel range: {img_array.min()} to {img_array.max()}")
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': 'No selected file'}), 400
    
    if not (file and allowed_file(file.filename)):
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Save the file temporarily
        file.save(filepath)
        logger.info(f"File saved temporarily at: {filepath}")
        
        # Preprocess the image
        processed_image = preprocess_image(filepath)
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = model.predict(processed_image)
        logger.info(f"Raw prediction output: {prediction}")
        
        # Process prediction results - CHANGED THIS SECTION
        probability = float(prediction[0][0])
        # Invert the probability if your model is giving opposite results
        inverted_probability = 1 - probability
        has_dysgraphia = inverted_probability > 0.5
        
        logger.info(f"""
        === PREDICTION RESULTS ===
        Raw Probability: {probability}
        Inverted Probability: {inverted_probability}
        Dysgraphia Detected: {has_dysgraphia}
        """)
        
        return jsonify({
            'probability': inverted_probability,  # Return the inverted probability
            'has_dysgraphia': has_dysgraphia,
            'message': 'Dysgraphia detected' if has_dysgraphia else 'No dysgraphia detected'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
        
    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Removed temporary file: {filepath}")
            except Exception as e:
                logger.error(f"Error removing file: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_input_shape': model.input_shape if model else None
    })

if __name__ == '__main__':
    # Print startup information
    logger.info("\n=== Starting Dysgraphia Detection API ===")
    logger.info(f"Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    logger.info(f"Allowed file types: {ALLOWED_EXTENSIONS}")
    
    if model:
        logger.info("\nModel Information:")
        logger.info(f"Input shape: {model.input_shape}")
        logger.info(f"Output shape: {model.output_shape}")
    else:
        logger.warning("Model failed to load - predictions will not work")
    
    if __name__ == "__main__":
        app.run()
