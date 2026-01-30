import os
import numpy as np
import pandas as pd
import pickle

def generate_dummy_data():
    print("Generating Dummy Data...")

    # 1. SECOM Data
    os.makedirs("data/raw_secom", exist_ok=True)
    n_samples = 100
    n_features = 590 # Standard SECOM features

    # Sensors (secom.data) - Space separated
    sensors = np.random.randn(n_samples, n_features)
    np.savetxt("data/raw_secom/secom.data", sensors, fmt='%.4f', delimiter=' ')

    # Labels (secom_labels.data) - Space separated: [Label, Timestamp]
    # Labels: -1 for Pass, 1 for Fail
    labels = np.random.choice([-1, 1], size=n_samples, p=[0.9, 0.1])
    timestamps = ["19/07/2008 11:55:00"] * n_samples

    with open("data/raw_secom/secom_labels.data", "w") as f:
        for l, t in zip(labels, timestamps):
            f.write(f"{l} {t}\n")

    print("- SECOM dummy data created.")

    # 2. WM-811K Data
    os.makedirs("data/raw_wm811k", exist_ok=True)
    n_wafers = 200 # Small number for speed

    wafer_data = []
    failure_types = [['Center'], ['Donut'], ['Edge-Loc'], ['Edge-Ring'], ['Loc'], ['Near-full'], ['Scratch'], ['Random'], []]

    for i in range(n_wafers):
        # waferMap: numpy array of 0, 1, 2 (0=background, 1=die, 2=defect)
        w_size = np.random.randint(20, 50)
        w_map = np.random.choice([0, 1, 2], size=(w_size, w_size), p=[0.1, 0.85, 0.05])

        f_type = failure_types[np.random.randint(len(failure_types))]

        wafer_data.append({
            'waferMap': w_map,
            'dieSize': w_size * w_size,
            'lotName': f"lot{i}",
            'waferIndex': i % 25,
            'trainTestLabel': [['Training']] if i % 2 == 0 else [['Test']],
            'failureType': f_type
        })

    df_wm = pd.DataFrame(wafer_data)
    with open("data/raw_wm811k/LSWMD.pkl", "wb") as f:
        pickle.dump(df_wm, f)

    print("- WM-811K dummy data created.")

    # 3. Create Processed Directory
    os.makedirs("data/processed", exist_ok=True)

    print("--- SUCCESS: All Dummy Data Generated! ---")

if __name__ == "__main__":
    generate_dummy_data()
