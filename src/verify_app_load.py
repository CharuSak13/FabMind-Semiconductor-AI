import os
import torch
import joblib
import numpy as np
from data_loader import FabMindDataLoader
from models import SensorAutoencoder, WaferMapCNN

def verify_load():
    print("Verifying model and data loading...")

    # Paths
    SECOM_PATH = "data/raw_secom/secom.data"
    LABELS_PATH = "data/raw_secom/secom_labels.data"
    PROCESSED_DIR = "data/processed"
    MODEL_DIR = "models"

    # 1. Load Data
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df_raw = loader.load_secom()
    df_clean = loader.clean_secom_data(df_raw)

    X_sensors = df_clean.drop(columns=['Pass_Fail']).values.astype(np.float32)
    y_secom = df_clean['Pass_Fail'].values
    print(f"Sensors loaded: {X_sensors.shape}")

    # 2. Load Images
    X_images = np.load(os.path.join(PROCESSED_DIR, "X_images_64.npy"))
    y_labels = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    print(f"Images loaded: {X_images.shape}")

    # 3. Load Models
    sensor_dim = X_sensors.shape[1]
    sensor_model = SensorAutoencoder(input_dim=sensor_dim, latent_dim=64)
    sensor_model.load_state_dict(torch.load(f"{MODEL_DIR}/sensor_autoencoder.pth"))
    sensor_model.eval()
    print("Sensor model loaded.")

    cnn_model = WaferMapCNN(latent_dim=64)
    cnn_model.load_state_dict(torch.load(f"{MODEL_DIR}/wafer_cnn.pth"), strict=False)
    cnn_model.eval()
    print("CNN model loaded.")

    xgb_model = joblib.load(f"{MODEL_DIR}/xgboost_yield.pkl")
    print("XGBoost model loaded.")

    # 4. Simple Inference Test
    s_tensor = torch.tensor(X_sensors[0]).unsqueeze(0)
    with torch.no_grad():
        s_emb = sensor_model.encoder(s_tensor)

    i_tensor = torch.tensor(X_images[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 2.0
    with torch.no_grad():
        i_emb = cnn_model(i_tensor)

    fused_vec = torch.cat((s_emb, i_emb), dim=1).numpy()
    prob = xgb_model.predict_proba(fused_vec)[0][1]
    print(f"Inference test successful. Failure Probability: {prob:.4f}")

    print("--- ALL SYSTEMS FUNCTIONAL ---")

if __name__ == "__main__":
    verify_load()
