# app.py

import os
import cv2
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import xgboost as xgb
from skimage import measure, feature
from scipy.stats import skew, kurtosis
from flask import Flask, request, jsonify, render_template
import base64
from PIL import Image
import io
import traceback
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- Load All Models and Components ONCE on Startup ---
print("--- Loading all trained models and components... ---")
try:
    UNET_MODEL_PATH = os.path.join("models", "Unet_model_final.h5")
    XGB_MODEL_PATH = os.path.join("models", "xgb_model_final.json")
    PIPELINE_COMPONENTS_PATH = os.path.join("models", "pipeline_components.pkl")

    unet_model = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    pipeline_components = joblib.load(PIPELINE_COMPONENTS_PATH)
    scaler = pipeline_components['scaler']
    label_encoder = pipeline_components['label_encoder']
    model_feature_names = pipeline_components['feature_names']
    
    print("--- All models loaded successfully. Ready to receive requests. ---")
except Exception as e:
    print(f"CRITICAL ERROR ON STARTUP: Could not load models. {e}")
    print(traceback.format_exc())

# --- Define Processing Functions from your Colab Notebook ---
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY); kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel); _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, thresh, 1, cv2.INPAINT_NS)

def extract_enhanced_features(image_rgb, pred_mask_prob):
    try:
        features = {}
        # Using the 0.3 threshold from your working Colab script
        mask_binary = (pred_mask_prob > 0.3).astype(np.uint8)
        props_list = measure.regionprops(mask_binary)
        if not props_list: return None, None
        props = props_list[0]
        if props.area < 20: return None, None
        
        area=props.area; perimeter=props.perimeter if props.perimeter>0 else 1
        
        features['asymmetry1']=props.inertia_tensor_eigvals[1]/(props.inertia_tensor_eigvals[0]+1e-6)
        features['asymmetry2']=props.minor_axis_length/(props.major_axis_length+1e-6)
        features['eccentricity']=props.eccentricity; features['solidity']=props.solidity
        features['border_irregularity1']=(perimeter**2)/(4*np.pi*area); features['compactness']=(4*np.pi*area)/(perimeter**2)
        lesion_pixels_rgb=image_rgb[mask_binary==1]
        if lesion_pixels_rgb.size == 0: return None, None
        for i,n in enumerate(['r','g','b']): c=lesion_pixels_rgb[:,i]; features[f'c_mean_{n}']=np.mean(c); features[f'c_std_{n}']=np.std(c)
        features['equiv_diam']=props.equivalent_diameter
        return features, mask_binary
    except Exception:
        return None, None

# --- Main Prediction Pipeline ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        original_image_rgb = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        IMG_SIZE = unet_model.input_shape[1]
        hairless_image = remove_hair(original_image_rgb)
        image_for_unet = cv2.resize(hairless_image, (IMG_SIZE, IMG_SIZE))
        image_for_unet_normalized = np.expand_dims(image_for_unet, axis=0) / 255.0
        
        predicted_mask_prob = unet_model.predict(image_for_unet_normalized, verbose=0)[0]
        predicted_mask_resized = cv2.resize(predicted_mask_prob, (original_image_rgb.shape[1], original_image_rgb.shape[0]))
        
        features, final_mask_array = extract_enhanced_features(original_image_rgb, predicted_mask_resized)
        
        if features is None:
            raise ValueError("Could not extract features. The U-Net model produced a blank mask.")

        features_df = pd.DataFrame([features])
        for col in model_feature_names:
            if col not in features_df.columns: features_df[col] = 0
        features_df = features_df[model_feature_names]
        
        scaled_features = scaler.transform(features_df)
        prediction_encoded = xgb_model.predict(scaled_features)
        prediction_proba = xgb_model.predict_proba(scaled_features)
        final_prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
        confidence_score = np.max(prediction_proba)
        
        prediction_text = f"Prediction: {final_prediction_label}\nConfidence: {confidence_score:.2%}"
        feature_text = (
            f"Asymmetry (Axis Ratio): {features.get('asymmetry2', 0):.4f}\n"
            f"Border (Compactness): {features.get('compactness', 0):.4f}\n"
            f"Color (Mean RGB): {features.get('c_mean_r', 0):.1f}, {features.get('c_mean_g', 0):.1f}, {features.get('c_mean_b', 0):.1f}\n"
            f"Diameter (Pixels): {features.get('equiv_diam', 0):.2f}"
        )
        
        final_mask_for_display = final_mask_array * 255
        
        final_mask_img = Image.fromarray(final_mask_for_display.astype(np.uint8))
        buff = io.BytesIO()
        final_mask_img.save(buff, format="PNG")
        mask_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        
        return jsonify({'prediction': prediction_text, 'features': feature_text, 'mask': mask_base64})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': f"An internal server error occurred: {e}"}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)