import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# MODEL 1: Sensor Autoencoder (For SECOM Data)
# ----------------------------------------
class SensorAutoencoder(nn.Module):
    def __init__(self, input_dim=563, latent_dim=64):
        super(SensorAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim), # This is the "Embedding"
            nn.ReLU()
        )
        # Decoder (Only needed for training, not for prediction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# ----------------------------------------
# MODEL 2: Wafer Map CNN (For WM-811K Images)
# ----------------------------------------
class WaferMapCNN(nn.Module):
    def __init__(self, latent_dim=64):
        super(WaferMapCNN, self).__init__()
        # Input is (1, 64, 64) -> Grayscale Image
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # 64x64 -> 32x32
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Pool: 32x32 -> 16x16
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Pool: 16x16 -> 8x8
        
        # Flatten
        self.flatten_size = 64 * 8 * 8
        self.fc = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # x shape: [batch, 64, 64] -> Needs [batch, 1, 64, 64]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, self.flatten_size) # Flatten
        x = self.fc(x) # Compress to latent_dim
        return x

# ----------------------------------------
# MODEL 3: The Fusion Brain (Combines Both)
# ----------------------------------------
class FabMindFusion(nn.Module):
    def __init__(self, sensor_dim=64, image_dim=64):
        super(FabMindFusion, self).__init__()
        
        # Combined Dimension
        fusion_input_dim = sensor_dim + image_dim 
        
        # Classification Head (Pass/Fail)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting
            nn.Linear(64, 1),
            nn.Sigmoid() # Output probability 0.0 to 1.0
        )

    def forward(self, sensor_embedding, image_embedding):
        # Concatenate Vectors
        fused = torch.cat((sensor_embedding, image_embedding), dim=1)
        output = self.classifier(fused)
        return output

# ----------------------------------------
# TEST CODE (Runs if you execute this file)
# ----------------------------------------
if __name__ == "__main__":
    print("Testing FabMind Architectures...")
    
    # 1. Test Sensor Model
    sensor_model = SensorAutoencoder()
    dummy_sensor = torch.randn(10, 563) # 10 samples
    enc, dec = sensor_model(dummy_sensor)
    print(f"Sensor Model: Encoded shape {enc.shape}") # Should be [10, 64]
    
    # 2. Test Image Model
    img_model = WaferMapCNN()
    dummy_img = torch.randn(10, 64, 64) # 10 images
    img_vec = img_model(dummy_img)
    print(f"Image Model: Encoded shape {img_vec.shape}") # Should be [10, 64]
    
    # 3. Test Fusion
    fusion_model = FabMindFusion()
    result = fusion_model(enc, img_vec)
    print(f"Fusion Model: Output shape {result.shape}") # Should be [10, 1]
    
    print("--- ALL SYSTEMS GO ---")