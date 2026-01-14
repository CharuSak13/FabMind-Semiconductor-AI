import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from data_loader import FabMindDataLoader
from models import SensorAutoencoder, WaferMapCNN

# CONFIGURATION
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Setup Styles for Professional Graphs
plt.style.use('seaborn-v0_8-whitegrid')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def generate_results():
    print("--- GENERATING PROFESSIONAL RESEARCH RESULTS ---")
    
    # ---------------------------------------------------------
    # 1. LOAD DATA & MODELS
    # ---------------------------------------------------------
    print("1. Loading Data & Models...")
    
    # Load Sensor Data
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df_raw = loader.load_secom()
    df_clean = loader.clean_secom_data(df_raw)
    
    y_secom = np.where(df_clean['Pass_Fail'].values == -1, 0, 1)
    X_sensors = df_clean.drop(columns=['Pass_Fail']).values.astype(np.float32)
    
    # Load Image Data
    X_images_all = np.load(os.path.join(PROCESSED_DIR, "X_images_64.npy"))
    y_labels_all = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    indices_good = np.where(y_labels_all == 'none')[0]
    indices_bad = np.where(y_labels_all != 'none')[0]
    
    # Load Models
    actual_sensor_dim = X_sensors.shape[1]
    sensor_model = SensorAutoencoder(input_dim=actual_sensor_dim, latent_dim=64)
    sensor_model.load_state_dict(torch.load(f"{MODEL_DIR}/sensor_autoencoder.pth"))
    sensor_model.eval()
    
    cnn_model = WaferMapCNN(latent_dim=64)
    cnn_model.load_state_dict(torch.load(f"{MODEL_DIR}/wafer_cnn.pth"), strict=False)
    cnn_model.eval()
    
    xgb_model = joblib.load(f"{MODEL_DIR}/xgboost_yield.pkl")

    # ---------------------------------------------------------
    # 2. GENERATE TEST SET VECTORS
    # ---------------------------------------------------------
    print("2. Generating Embeddings for Evaluation...")
    
    sensor_vecs = []
    image_vecs = []
    fused_vecs = []
    
    with torch.no_grad():
        for i in range(len(X_sensors)):
            # Sensor
            s_tensor = torch.tensor(X_sensors[i]).unsqueeze(0)
            s_emb = sensor_model.encoder(s_tensor)
            
            # Image (Matching Logic)
            if y_secom[i] == 1: idx = np.random.choice(indices_bad)
            else: idx = np.random.choice(indices_good)
            
            img_data = X_images_all[idx]
            i_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0) / 2.0
            if len(i_tensor.shape) == 3: i_tensor = i_tensor.unsqueeze(1)
            i_emb = cnn_model(i_tensor)
            
            # Store
            s_numpy = s_emb.numpy().flatten()
            i_numpy = i_emb.numpy().flatten()
            f_numpy = np.concatenate([s_numpy, i_numpy])
            
            sensor_vecs.append(s_numpy)
            image_vecs.append(i_numpy)
            fused_vecs.append(f_numpy)
            
    X_fused = np.array(fused_vecs)
    y_true = y_secom
    
    # Split to get a Test Set
    X_train, X_test, y_train, y_test = train_test_split(X_fused, y_true, test_size=0.2, random_state=42)
    
    # Predictions
    y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = xgb_model.predict(X_test)
    
    # ---------------------------------------------------------
    # 3. GRAPH 1: CONFUSION MATRIX
    # ---------------------------------------------------------
    print("3. Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pass', 'Fail'], yticklabels=['Pass', 'Fail'])
    plt.title('Confusion Matrix: FabMind Fusion Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{RESULTS_DIR}/1_confusion_matrix.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 4. GRAPH 2: ROC-AUC CURVE
    # ---------------------------------------------------------
    print("4. Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f"{RESULTS_DIR}/2_roc_curve.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 5. GRAPH 3: GLOBAL FEATURE IMPORTANCE (SHAP)
    # ---------------------------------------------------------
    print("5. Generating Global SHAP Summary...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, max_display=15, show=False, plot_type="bar")
    plt.title("Top 15 Features Driving Yield Predictions (Global)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/3_shap_global.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 6. GRAPH 4: ABLATION STUDY (The Fix is Here)
    # ---------------------------------------------------------
    print("6. Generating SOTA Comparison (Ablation Study)...")
    
    # Model A: Sensors Only
    # FIX: We now define y_train_s and y_test_s explicitly
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(np.array(sensor_vecs), y_true, test_size=0.2, random_state=42)
    model_s = xgb.XGBClassifier(scale_pos_weight=10, eval_metric='logloss')
    model_s.fit(X_train_s, y_train_s) 
    acc_sensor = accuracy_score(y_test_s, model_s.predict(X_test_s))
    
    # Model B: Images Only
    # FIX: We now define y_train_i and y_test_i explicitly
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(np.array(image_vecs), y_true, test_size=0.2, random_state=42)
    model_i = xgb.XGBClassifier(scale_pos_weight=10, eval_metric='logloss')
    model_i.fit(X_train_i, y_train_i)
    acc_image = accuracy_score(y_test_i, model_i.predict(X_test_i))
    
    # Model C: Fusion (Already calculated)
    acc_fusion = accuracy_score(y_test, y_pred)
    
    # Plotting
    methods = ['Sensors Only', 'Images Only', 'FabMind Fusion']
    accuracies = [acc_sensor, acc_image, acc_fusion]
    colors = ['gray', 'gray', '#00ff41']
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, accuracies, color=colors)
    plt.ylim(0.8, 1.0)
    plt.title('Performance Comparison: Unimodal vs. Multimodal')
    plt.ylabel('Accuracy')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval*100:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.savefig(f"{RESULTS_DIR}/4_ablation_comparison.png", dpi=300)
    plt.close()
    
    print("\n--- DONE! CHECK THE 'results' FOLDER ---")

if __name__ == "__main__":
    generate_results()