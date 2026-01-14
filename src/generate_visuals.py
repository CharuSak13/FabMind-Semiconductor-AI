import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.manifold import TSNE
from data_loader import FabMindDataLoader
from models import SensorAutoencoder, WaferMapCNN
from sklearn.model_selection import train_test_split

# CONFIGURATION
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

plt.style.use('seaborn-v0_8-white')

def generate_advanced_visuals():
    print("--- GENERATING ADVANCED SOTA VISUALS ---")
    
    # 1. LOAD EVERYTHING (Same as before)
    print("1. Loading System...")
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df_raw = loader.load_secom()
    df_clean = loader.clean_secom_data(df_raw)
    y_secom = np.where(df_clean['Pass_Fail'].values == -1, 0, 1)
    X_sensors = df_clean.drop(columns=['Pass_Fail']).values.astype(np.float32)
    
    X_images_all = np.load(os.path.join(PROCESSED_DIR, "X_images_64.npy"))
    y_labels_all = np.load(os.path.join(PROCESSED_DIR, "y_labels.npy"))
    indices_good = np.where(y_labels_all == 'none')[0]
    indices_bad = np.where(y_labels_all != 'none')[0]
    
    # Load Models
    actual_dim = X_sensors.shape[1]
    sensor_model = SensorAutoencoder(input_dim=actual_dim, latent_dim=64)
    sensor_model.load_state_dict(torch.load(f"{MODEL_DIR}/sensor_autoencoder.pth"))
    sensor_model.eval()
    
    cnn_model = WaferMapCNN(latent_dim=64)
    cnn_model.load_state_dict(torch.load(f"{MODEL_DIR}/wafer_cnn.pth"), strict=False)
    cnn_model.eval()
    
    xgb_model = joblib.load(f"{MODEL_DIR}/xgboost_yield.pkl")

    # 2. GENERATE VECTORS (We need a batch of data)
    print("2. Processing Batch for Visualization...")
    fused_vecs = []
    images_used = []
    labels_used = [] # Actual labels
    preds_used = []  # Model predictions
    probs_used = []  # Model confidence
    
    # We will process 500 samples for the t-SNE plot, and pick 10 for the Gallery
    sample_size = 500
    
    with torch.no_grad():
        for i in range(sample_size):
            # Process Sensor
            s_tensor = torch.tensor(X_sensors[i]).unsqueeze(0)
            s_emb = sensor_model.encoder(s_tensor)
            
            # Process Image
            if y_secom[i] == 1: idx = np.random.choice(indices_bad)
            else: idx = np.random.choice(indices_good)
            
            img_raw = X_images_all[idx]
            defect_name = y_labels_all[idx]
            
            i_tensor = torch.tensor(img_raw, dtype=torch.float32).unsqueeze(0) / 2.0
            if len(i_tensor.shape) == 3: i_tensor = i_tensor.unsqueeze(1)
            i_emb = cnn_model(i_tensor)
            
            # Fuse
            fused = torch.cat((s_emb, i_emb), dim=1).numpy().flatten()
            fused_vecs.append(fused)
            images_used.append(img_raw)
            labels_used.append(y_secom[i]) # 0 or 1
            
            # Predict immediately
            prob = xgb_model.predict_proba([fused])[0][1]
            pred = 1 if prob > 0.5 else 0
            
            preds_used.append(pred)
            probs_used.append(prob)

    X_embedded = np.array(fused_vecs)
    
    # ---------------------------------------------------------
    # VISUAL 1: PREDICTION GALLERY (The "Wafer with Label" Request)
    # ---------------------------------------------------------
    print("3. Generating Prediction Gallery...")
    
    # We want to show a mix of Pass and Fail
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle('FabMind Real-Time Defect Detection Gallery', fontsize=16)
    
    # Pick 10 random indices from our processed batch
    gallery_indices = np.random.choice(sample_size, 10, replace=False)
    
    for i, ax in enumerate(axes.flat):
        idx = gallery_indices[i]
        img = images_used[idx]
        actual = labels_used[idx]
        pred = preds_used[idx]
        conf = probs_used[idx]
        
        # Color code title: Green if Correct, Red if Wrong
        color = 'green' if actual == pred else 'red'
        status = "FAIL" if pred == 1 else "PASS"
        confidence = conf if pred == 1 else 1-conf
        
        ax.imshow(img, cmap='inferno')
        ax.set_title(f"Pred: {status} ({confidence:.1%})\nActual: {'FAIL' if actual==1 else 'PASS'}", color=color, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/5_prediction_gallery.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # VISUAL 2: t-SNE CLUSTER PLOT (The "Science" Proof)
    # ---------------------------------------------------------
    print("4. Generating t-SNE Latent Space Visualization...")
    
    # t-SNE reduces 128 dimensions -> 2 dimensions for plotting
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_embedded)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Passes (Blue)
    plt.scatter(X_tsne[np.array(labels_used)==0, 0], X_tsne[np.array(labels_used)==0, 1], 
                c='dodgerblue', label='Pass (Good)', alpha=0.6, s=50)
    
    # Plot Fails (Red)
    plt.scatter(X_tsne[np.array(labels_used)==1, 0], X_tsne[np.array(labels_used)==1, 1], 
                c='red', label='Fail (Defect)', alpha=0.8, s=50, edgecolors='black')
    
    plt.title("FabMind Latent Space: Visualizing Decision Boundaries", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(f"{RESULTS_DIR}/6_tsne_clusters.png", dpi=300)
    plt.close()
    
    print("\n--- NEW VISUALS GENERATED IN 'results/' FOLDER ---")

if __name__ == "__main__":
    generate_advanced_visuals()