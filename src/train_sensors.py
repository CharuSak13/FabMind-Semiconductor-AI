import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import FabMindDataLoader
from models import SensorAutoencoder
import os
import numpy as np

# Configuration
SECOM_PATH = "data/raw_secom/secom.data"
LABELS_PATH = "data/raw_secom/secom_labels.data"
MODEL_SAVE_PATH = "models/sensor_autoencoder.pth"
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def train_sensor_model():
    print("--- Starting Phase 4A: Training Sensor Autoencoder ---")
    
    # 1. Load Data
    # We pass None for wm811k because we don't need images right now
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, None)
    df_raw = loader.load_secom()
    df_clean = loader.clean_secom_data(df_raw)
    
    # Drop "Pass_Fail" column, we only train on sensors (Unsupervised)
    X_data = df_clean.drop(columns=['Pass_Fail']).values.astype(np.float32)
    
    # Convert to PyTorch Tensor
    tensor_x = torch.tensor(X_data)
    
    # 2. Setup Model
    input_dim = X_data.shape[1] # Should be 563
    print(f"Input Features: {input_dim}")
    
    model = SensorAutoencoder(input_dim=input_dim, latent_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # We want Output to match Input (Reconstruction)
    
    # 3. Training Loop
    model.train()
    print(f"Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        # Forward pass
        encoded, decoded = model(tensor_x)
        
        # Calculate Loss (How bad is the reconstruction?)
        loss = criterion(decoded, tensor_x)
        
        # Backward pass (Learn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    # 4. Save the Brain
    if not os.path.exists("models"):
        os.makedirs("models")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"--- SUCCESS: Sensor Model Saved to {MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    train_sensor_model()