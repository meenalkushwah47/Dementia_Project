from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import pandas as pd
import traceback
import os

app = Flask(_name_)
CORS(app)

# Global variables
model = None
label_encoders = None
feature_names = None

# Load model and encoders
def load_model_files():
    global model, label_encoders, feature_names
    
    print("\nChecking for required files...")
    required_files = ['dementia_model.pkl', 'label_encoders.pkl', 'feature_names.pkl']
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ {file} NOT FOUND!")
            return False
        print(f"✓ {file} found")
    
    try:
        with open('dementia_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")

        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        print("✓ Label encoders loaded successfully")

        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✓ Feature names loaded: {len(feature_names)} features")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model files: {e}")
        traceback.print_exc()
        return False

# Load files at startup
if not load_model_files():
    print("\n" + "="*60)
    print("WARNING: Model files not found!")
    print("="*60 + "\n")

@app.route('/', methods=['GET'])
def home():
    return send_file('main.html')

@app.route('/main.css', methods=['GET'])
def css():
    return send_file('main.css')

@app.route('/main.js', methods=['GET'])
def js():
    return send_file('main.js')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    print(f"\n{'='*60}")
    print(f"REQUEST RECEIVED: {request.method} {request.url}")
    print(f"Content-Type: {request.headers.get('Content-Type')}")
    print(f"{'='*60}")
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    print("Processing POST request...")
    
    try:
        # Check if model is loaded
        if model is None or label_encoders is None or feature_names is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'No data provided'}), 400
        
        print(f"Received data: {list(data.keys())}")
        
        input_data = {}
        
        # Numeric features
        input_data['Diabetic'] = int(data.get('Diabetic', 0))
        input_data['AlcoholLevel'] = float(data.get('AlcoholLevel', 0))
        input_data['HeartRate'] = int(data.get('HeartRate', 0))
        input_data['BloodOxygenLevel'] = float(data.get('BloodOxygenLevel', 0))
        input_data['BodyTemperature'] = float(data.get('BodyTemperature', 0))
        input_data['Weight'] = float(data.get('Weight', 0))
        input_data['MRI_Delay'] = float(data.get('MRI_Delay', 0))
        input_data['Age'] = int(data.get('Age', 0))
        input_data['Cognitive_Test_Scores'] = int(data.get('Cognitive_Test_Scores', 0))
        input_data['Dosage in mg'] = float(data.get('Dosage_in_mg', 0))
        
        # Categorical features
        categorical_features = {
            'Prescription': data.get('Prescription', 'None'),
            'Education_Level': data.get('Education_Level', 'None'),
            'Dominant_Hand': data.get('Dominant_Hand', 'Right'),
            'Gender': data.get('Gender', 'Male'),
            'Family_History': data.get('Family_History', 'No'),
            'Smoking_Status': data.get('Smoking_Status', 'Never Smoked'),
            'APOE_ε4': data.get('APOE_ε4', 'Negative'),
            'Physical_Activity': data.get('Physical_Activity', 'Sedentary'),
            'Depression_Status': data.get('Depression_Status', 'No'),
            'Medication_History': data.get('Medication_History', 'No'),
            'Nutrition_Diet': data.get('Nutrition_Diet', 'Balanced Diet'),
            'Sleep_Quality': data.get('Sleep_Quality', 'Good'),
            'Chronic_Health_Conditions': data.get('Chronic_Health_Conditions', 'None')
        }
        
        # Encode categorical features
        for feature, value in categorical_features.items():
            if feature in label_encoders:
                try:
                    input_data[feature] = label_encoders[feature].transform([value])[0]
                except:
                    input_data[feature] = 0
            else:
                input_data[feature] = value
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        df = df[feature_names]
        
        print(f"Input shape: {df.shape}")
        
        # Predict
        prediction = model.predict(df)[0]
        confidence = None
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            confidence = float(max(proba))
        
        result = {
            'prediction': int(prediction),
            'result_text': 'Yes - Dementia Detected' if prediction == 1 else 'No - No Dementia Detected',
            'confidence': confidence
        }
        
        print(f"Prediction: {result}")
        print(f"{'='*60}\n")
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if _name_ == '_main_':
    print("\n" + "="*60)
    print("DEMENTIA PREDICTION SERVER")
    print("="*60)
    print("Starting server...")
    print("="*60 + "\n")
    
    # Use PORT from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0',port=port