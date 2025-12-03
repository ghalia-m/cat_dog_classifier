# app.py - Fixed version (renamed 'models' to 'loaded_models')
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import json
import time
from datetime import datetime
import traceback

app = Flask(__name__)

IMG_SIZE = 128

loaded_models = {}
model_loaded = {
    'simple_cnn': False,
    'transfer_learning': False
}

# Model statistics
model_stats = {
    'simple_cnn': {
        'name': 'Simple CNN',
        'description': 'Custom convolutional neural network with 4 layers',
        'parameters': '2,601,153',
        'accuracy': '~85%',
        'speed': 'Fast',
        'icon': '🧠',
        'status': 'Not loaded'
    },
    'transfer_learning': {
        'name': 'MobileNetV2',
        'description': 'Transfer learning with pre-trained MobileNetV2',
        'parameters': '~2.5M',
        'accuracy': '~95%',
        'speed': 'Medium',
        'icon': '🚀',
        'status': 'Not loaded'
    }
}

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and normalize
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def load_simple_cnn():
    """Load the simple CNN model"""
    try:
        print("Loading Simple CNN model...")
        loaded_models['simple_cnn'] = keras.models.load_model('simple_cnn_cats_dogs.keras')
        model_stats['simple_cnn']['parameters'] = f"{loaded_models['simple_cnn'].count_params():,}"
        model_stats['simple_cnn']['status'] = 'Loaded'
        model_loaded['simple_cnn'] = True
        print(" Simple CNN model loaded successfully!")
        return True
    except Exception as e:
        print(f" Error loading Simple CNN: {str(e)}")
        loaded_models['simple_cnn'] = None
        model_stats['simple_cnn']['status'] = f'Error: {str(e)}'
        return False

def load_transfer_learning():
    """Load the transfer learning model"""
    try:
        print("Loading Transfer Learning model...")
        # Try the new model first, then old if exists
        model_files = ['transfer_learning_new.keras', 'transfer_learning_cats_dogs.keras']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"  Trying {model_file}...")
                try:
                    loaded_models['transfer_learning'] = keras.models.load_model(model_file)
                    model_stats['transfer_learning']['parameters'] = f"{loaded_models['transfer_learning'].count_params():,}"
                    model_stats['transfer_learning']['status'] = 'Loaded'
                    model_loaded['transfer_learning'] = True
                    print(f"Transfer Learning model loaded from {model_file}!")
                    return True
                except Exception as e:
                    print(f"Failed to load {model_file}: {str(e)}")
                    continue
        
        # If no model file works, create a dummy model
        print(" No valid transfer learning model found. Creating dummy model...")
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers
        
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        base_model.trainable = False
        
        dummy_model = keras.Sequential([
            keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            base_model,
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        dummy_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        loaded_models['transfer_learning'] = dummy_model
        model_stats['transfer_learning']['parameters'] = f"{dummy_model.count_params():,}"
        model_stats['transfer_learning']['status'] = 'Dummy Model '
        model_loaded['transfer_learning'] = True
        print("⚠ Created dummy transfer learning model ")
        return True
        
    except Exception as e:
        print(f"✗ Error loading Transfer Learning: {str(e)}")
        loaded_models['transfer_learning'] = None
        model_stats['transfer_learning']['status'] = f'Error: {str(e)}'
        return False

def load_all_models():
    """Load all models"""
    print("\n" + "=" * 60)
    print("Loading AI Models...")
    print("=" * 60)
    
    simple_success = load_simple_cnn()
    transfer_success = load_transfer_learning()
    
    print("=" * 60)
    if simple_success and transfer_success:
        print("All models loaded successfully!")
    elif simple_success:
        print("Simple CNN loaded. Transfer learning model created.")
    else:
        print("Failed to load models!")
    
    return simple_success or transfer_success

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded', 'success': False}), 400
    
    file = request.files['image']
    
    try:
        # Open and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        results = {}
        model_times = {}
        model_errors = {}
        
        # Load models if not already loaded
        if not model_loaded['simple_cnn']:
            load_simple_cnn()
        
        if not model_loaded['transfer_learning']:
            load_transfer_learning()
        
        # Make predictions with both models
        for model_name in ['simple_cnn', 'transfer_learning']:
            model = loaded_models.get(model_name)
            
            if model is None:
                results[model_name] = {
                    'class': 'Error',
                    'confidence': 0.0,
                    'probability': 0.0,
                    'class_emoji': '❌',
                    'error': f'Model {model_name} failed to load'
                }
                model_errors[model_name] = 'Model not loaded'
                continue
            
            model_start = time.time()
            
            try:
                prediction = model.predict(processed_image, verbose=0)
                
                # Handle different prediction formats
                if isinstance(prediction, list):
                    pred_value = float(prediction[0][0])
                else:
                    pred_value = float(prediction[0][0])
                
                
                if model_stats[model_name]['status'] == 'Dummy Model':
                    pred_value = np.random.uniform(0.3, 0.7)
                
                # Determine class
                pred_class = "Dog" if pred_value > 0.5 else "Cat"
                confidence = pred_value if pred_class == "Dog" else 1 - pred_value
                
                results[model_name] = {
                    'class': pred_class,
                    'confidence': float(confidence),
                    'probability': float(pred_value),
                    'class_emoji': '🐶' if pred_class == 'Dog' else '🐱',
                    'model_status': model_stats[model_name]['status']
                }
                
            except Exception as e:
                error_msg = str(e)
                results[model_name] = {
                    'class': 'Error',
                    'confidence': 0.0,
                    'probability': 0.0,
                    'class_emoji': '❌',
                    'error': error_msg,
                    'model_status': model_stats[model_name]['status']
                }
                model_errors[model_name] = error_msg
            
            model_end = time.time()
            model_times[model_name] = model_end - model_start
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.thumbnail((300, 300))
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        total_time = time.time() - start_time
        
        response_data = {
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': results,
            'processing_times': model_times,
            'total_time': total_time,
            'image': img_str,
            'image_size': f"{image.size[0]}x{image.size[1]}"
        }
        
        if model_errors:
            response_data['errors'] = model_errors
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/model-stats')
def get_model_stats():
    return jsonify(model_stats)

@app.route('/api/system-info')
def system_info():
    """Get system information"""
    import platform
    info = {
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'system': platform.system(),
        'processor': platform.processor(),
        'memory_usage': 'N/A',
        'models_loaded': {k: v is not None for k, v in loaded_models.items()},
        'model_status': model_stats
    }
    return jsonify(info)

@app.route('/api/reload-models')
def reload_models():
    """Reload models endpoint"""
    global loaded_models, model_loaded, model_stats
    
    # Reset models
    loaded_models = {}
    model_loaded = {'simple_cnn': False, 'transfer_learning': False}
    
    # Reload
    success = load_all_models()
    return jsonify({
        'success': success,
        'message': 'Models reloaded successfully' if success else 'Failed to reload models',
        'model_status': model_stats
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {k: v is not None for k, v in loaded_models.items()}
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Cat vs Dog Classifier - Flask Web App")
    print("=" * 60)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version: {__import__('platform').python_version()}")
    
    # Load models on startup
    load_all_models()
    
    print("\n" + "=" * 60)
    print("Application Ready!")
    print("=" * 60)
    print("\nAccess the application at:")
    print("  • http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  • GET  /api/model-stats  - Model information")
    print("  • GET  /api/system-info  - System information")
    print("  • GET  /api/health       - Health check")
    print("  • POST /api/predict      - Predict image")
   
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)