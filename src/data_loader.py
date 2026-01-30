import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

class FabMindDataLoader:
    def __init__(self, data_path_secom, data_path_labels, data_path_wm811k):
        self.secom_path = data_path_secom
        self.labels_path = data_path_labels
        self.wm811k_path = data_path_wm811k
        
    def load_secom(self):
        """Phase 1A: Load Sensor Data"""
        print("Loading SECOM Data...")
        if not os.path.exists(self.secom_path):
            raise FileNotFoundError(f"Missing: {self.secom_path}")
            
        df_sensors = pd.read_csv(self.secom_path, sep=" ", header=None)
        df_sensors = df_sensors.dropna(axis=1, how='all')
        
        df_labels = pd.read_csv(self.labels_path, sep=r"\s+", header=None, engine='python')
        if df_labels.shape[1] == 3:
            df_labels[1] = df_labels[1] + " " + df_labels[2]
            df_labels = df_labels.drop(columns=[2])
        df_labels.columns = ['Pass_Fail', 'Timestamp']
        
        df = pd.concat([df_labels, df_sensors], axis=1)
        # Fix for the warning you saw earlier
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
        
        print(f"SECOM Data Loaded. Shape: {df.shape}")
        return df

    def clean_secom_data(self, df):
        """Phase 1B: Clean Sensor Data"""
        print("Cleaning SECOM Data...")
        labels = df['Pass_Fail']
        sensors = df.drop(['Pass_Fail', 'Timestamp'], axis=1)
        
        # Drop empty columns
        threshold = len(sensors) * 0.5
        sensors_dropped = sensors.dropna(thresh=threshold, axis=1)
        
        # Fill missing values
        imputer = SimpleImputer(strategy='mean')
        sensors_imputed = pd.DataFrame(imputer.fit_transform(sensors_dropped))
        sensors_imputed.columns = sensors_dropped.columns
        
        # Scale values
        scaler = StandardScaler()
        sensors_scaled = pd.DataFrame(scaler.fit_transform(sensors_imputed))
        
        df_clean = pd.concat([labels.reset_index(drop=True), sensors_scaled], axis=1)
        print("SECOM Cleaning Complete.")
        return df_clean

    def load_wm811k(self):
        """Phase 1C: Load Image Data (The Heavy Lifter)"""
        print(f"Loading WM-811K from {self.wm811k_path}...")
        print("This might take 10-20 seconds (It's 2GB)...")
        
        if not os.path.exists(self.wm811k_path):
             raise FileNotFoundError(f"Missing: {self.wm811k_path}")

        # Load Pickle
        df_wm = pd.read_pickle(self.wm811k_path)
        
        # Inspect columns
        print(f"WM-811K Loaded. Shape: {df_wm.shape}")
        
        # Filter: We only want rows that have failure labels for now
        # The dataset has columns: ['waferMap', 'dieSize', 'lotName', 'waferIndex', 'trianTestLabel', 'failureType']
        return df_wm

if __name__ == "__main__":
    # PATHS
    SECOM_PATH = "data/raw_secom/secom.data"
    LABELS_PATH = "data/raw_secom/secom_labels.data"
    WM811K_PATH = "data/raw_wm811k/LSWMD.pkl" # Make sure this matches your filename!
    
    loader = FabMindDataLoader(SECOM_PATH, LABELS_PATH, WM811K_PATH)
    
    # 1. Test SECOM
    df_raw = loader.load_secom()
    df_clean = loader.clean_secom_data(df_raw)
    
    # 2. Test WM-811K (New Step)
    df_images = loader.load_wm811k()
    
    print("\n--- SUCCESS ---")
    print(f"Sensors Ready: {df_clean.shape}")
    print(f"Images Ready: {df_images.shape}")
    print("Phase 1 Data Engineering Complete!")