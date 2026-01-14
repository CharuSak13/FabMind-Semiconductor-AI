import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import joblib
from data_loader import FabMindDataLoader
from models import SensorAutoencoder, WaferMapCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Configuration
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

def train_fusion_model():
    print("--- Starting Phase 5: Multimodal Fusion & Yield Prediction ---")
    
    # ---------------------------------------------------------
    # 1. PREPARE DATA FIRST (To get correct shapes)
    # ---------------------------------------------------------
    print("Preparing Fusion Dataset...")
    
    # Load SECOM (Sensors)
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df_raw = loader.load_secom()
    df_clean = loader.clean_secom_data(df_raw)
    
    # Separate Inputs and Labels
    y_secom = df_clean['Pass_Fail'].values
    y_secom = np.where(y_secom == -1, 0, 1) # 0=Pass, 1=Fail
    
    # Drop Label to get Features
    X_sensors = df_clean.drop(columns=['Pass_Fail']).values.astype(np.float32)
    
    # DYNAMIC DIMENSION CHECK (The Fix)
    actual_sensor_dim = X_sensors.shape[1]
    print(f"Detected Sensor Features: {actual_sensor_dim}") 
    
    # Load Image Data Source
    print("Loading Image Data Source...")
    X_images_all = np.load(os.path.join(PROCESSED_DIR, "X_images_64.npy"))
    y_labels_all = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    
    indices_good = np.where(y_labels_all == 'none')[0]
    indices_bad = np.where(y_labels_all != 'none')[0]
    
    # ---------------------------------------------------------
    # 2. LOAD PRE-TRAINED MODELS
    # ---------------------------------------------------------
    print("Loading Pre-Trained AI Models...")
    
    # Load Sensor Model (Using the detected dimension!)
    sensor_model = SensorAutoencoder(input_dim=actual_sensor_dim, latent_dim=64)
    sensor_model.load_state_dict(torch.load(f"{MODEL_DIR}/sensor_autoencoder.pth"))
    sensor_model.eval()
    
    # Load Image Model
    cnn_model = WaferMapCNN(latent_dim=64)
    cnn_model.load_state_dict(torch.load(f"{MODEL_DIR}/wafer_cnn.pth"), strict=False)
    cnn_model.eval()
    
    print("Models Loaded Successfully.")

    # ---------------------------------------------------------
    # 3. GENERATE FUSED EMBEDDINGS
    # ---------------------------------------------------------
    print("Generating Fused Vectors (This mimics the production line)...")
    
    fused_vectors = []
    final_labels = []
    
    with torch.no_grad():
        for i in range(len(X_sensors)):
            # A. Sensor Embedding
            sensor_tensor = torch.tensor(X_sensors[i]).unsqueeze(0)
            sensor_emb = sensor_model.encoder(sensor_tensor) 
            
            # B. Pick Matching Image
            is_fail = y_secom[i] == 1
            if is_fail:
                idx = np.random.choice(indices_bad)
            else:
                idx = np.random.choice(indices_good)
            
            img_data = X_images_all[idx]
            
            # C. Image Embedding
            img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0) / 2.0
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(1)
            
            img_emb = cnn_model(img_tensor) 
            
            # D. Fuse
            fused = torch.cat((sensor_emb, img_emb), dim=1).numpy().flatten()
            
            fused_vectors.append(fused)
            final_labels.append(is_fail)
            
            if i % 200 == 0:
                print(f"\rProcessed {i}/{len(X_sensors)}...", end="")

    print("\nFusion Complete.")
    X_fused = np.array(fused_vectors)
    y_fused = np.array(final_labels)
    
    # ---------------------------------------------------------
    # 4. TRAIN XGBOOST
    # ---------------------------------------------------------
    print("Training XGBoost Yield Predictor...")
    
    X_train, X_test, y_train, y_test = train_test_split(X_fused, y_fused, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=10, 
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 5. EVALUATE & SAVE
    # ---------------------------------------------------------
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print("\n--- RESULTS ---")
    print(f"Yield Prediction Accuracy: {acc*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=['Pass', 'Fail']))
    
    joblib.dump(model, f"{MODEL_DIR}/xgboost_yield.pkl")
    print(f"--- SUCCESS: Fusion Model Saved to {MODEL_DIR}/xgboost_yield.pkl ---")

if __name__ == "__main__":
    train_fusion_model()