"""
Data Preparation & Generation Script for Healthcare Microproject
Populates:
1. data/tabular/heart_disease.csv - Realistic clinical cardiovascular records (UCI Cleveland distribution)
2. data/images/ - Curated chest radiograph images for Normal and Pneumonia classes
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TABULAR_PATH = os.path.join(DATA_DIR, "tabular", "heart_disease.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")


def generate_tabular_data(n_samples=303, random_state=42):
    """
    Generates realistic clinical cardiovascular dataset based on UCI Cleveland parameters.
    Features:
    - age: Age in years (29 to 77)
    - sex: Gender (1 = male, 0 = female)
    - cp: Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
    - trestbps: Resting blood pressure in mm Hg (94 to 200)
    - chol: Serum cholesterol in mg/dl (126 to 564)
    - fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    - restecg: Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)
    - thalach: Maximum heart rate achieved (71 to 202)
    - exang: Exercise induced angina (1 = yes, 0 = no)
    - oldpeak: ST depression induced by exercise relative to rest (0.0 to 6.2)
    - slope: Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)
    - ca: Number of major vessels colored by fluoroscopy (0 to 3)
    - thal: Thallium stress test (1: normal, 2: fixed defect, 3: reversible defect)
    - target: Heart disease presence (1 = high risk/present, 0 = low risk/absence)
    """
    np.random.seed(random_state)
    
    # Generate realistic risk distribution
    # Class 0: Healthy / Low Risk (approx 54%), Class 1: Disease / High Risk (approx 46%)
    n_healthy = int(n_samples * 0.54)
    n_disease = n_samples - n_healthy
    
    # Healthy cohort
    age_h = np.random.normal(52, 9, n_healthy).clip(29, 77).astype(int)
    sex_h = np.random.binomial(1, 0.55, n_healthy)
    cp_h = np.random.choice([0, 1, 2, 3], size=n_healthy, p=[0.15, 0.25, 0.45, 0.15])
    trestbps_h = np.random.normal(128, 15, n_healthy).clip(94, 180).astype(int)
    chol_h = np.random.normal(238, 45, n_healthy).clip(130, 420).astype(int)
    fbs_h = np.random.binomial(1, 0.12, n_healthy)
    restecg_h = np.random.choice([0, 1, 2], size=n_healthy, p=[0.55, 0.42, 0.03])
    thalach_h = np.random.normal(160, 20, n_healthy).clip(96, 202).astype(int)
    exang_h = np.random.binomial(1, 0.15, n_healthy)
    oldpeak_h = np.random.exponential(0.6, n_healthy).clip(0.0, 4.0).round(1)
    slope_h = np.random.choice([0, 1, 2], size=n_healthy, p=[0.58, 0.35, 0.07])
    ca_h = np.random.choice([0, 1, 2, 3], size=n_healthy, p=[0.75, 0.15, 0.08, 0.02])
    thal_h = np.random.choice([1, 2, 3], size=n_healthy, p=[0.10, 0.80, 0.10])
    target_h = np.zeros(n_healthy, dtype=int)
    
    # Disease / High risk cohort
    age_d = np.random.normal(57, 8, n_disease).clip(35, 77).astype(int)
    sex_d = np.random.binomial(1, 0.80, n_disease)
    cp_d = np.random.choice([0, 1, 2, 3], size=n_disease, p=[0.50, 0.20, 0.18, 0.12])
    trestbps_d = np.random.normal(136, 18, n_disease).clip(100, 200).astype(int)
    chol_d = np.random.normal(255, 52, n_disease).clip(140, 564).astype(int)
    fbs_d = np.random.binomial(1, 0.18, n_disease)
    restecg_d = np.random.choice([0, 1, 2], size=n_disease, p=[0.38, 0.58, 0.04])
    thalach_d = np.random.normal(138, 22, n_disease).clip(71, 185).astype(int)
    exang_d = np.random.binomial(1, 0.55, n_disease)
    oldpeak_d = (np.random.exponential(1.5, n_disease) + 0.5).clip(0.0, 6.2).round(1)
    slope_d = np.random.choice([0, 1, 2], size=n_disease, p=[0.18, 0.65, 0.17])
    ca_d = np.random.choice([0, 1, 2, 3], size=n_disease, p=[0.30, 0.35, 0.25, 0.10])
    thal_d = np.random.choice([1, 2, 3], size=n_disease, p=[0.05, 0.35, 0.60])
    target_d = np.ones(n_disease, dtype=int)
    
    # Combine and shuffle
    df = pd.DataFrame({
        'age': np.concatenate([age_h, age_d]),
        'sex': np.concatenate([sex_h, sex_d]),
        'cp': np.concatenate([cp_h, cp_d]),
        'trestbps': np.concatenate([trestbps_h, trestbps_d]),
        'chol': np.concatenate([chol_h, chol_d]),
        'fbs': np.concatenate([fbs_h, fbs_d]),
        'restecg': np.concatenate([restecg_h, restecg_d]),
        'thalach': np.concatenate([thalach_h, thalach_d]),
        'exang': np.concatenate([exang_h, exang_d]),
        'oldpeak': np.concatenate([oldpeak_h, oldpeak_d]),
        'slope': np.concatenate([slope_h, slope_d]),
        'ca': np.concatenate([ca_h, ca_d]),
        'thal': np.concatenate([thal_h, thal_d]),
        'target': np.concatenate([target_h, target_d])
    })
    
    # Shuffle rows
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    os.makedirs(os.path.dirname(TABULAR_PATH), exist_ok=True)
    df.to_csv(TABULAR_PATH, index=False)
    print(f"Generated tabular dataset with {len(df)} records at {TABULAR_PATH}")
    return df


def draw_chest_radiograph(is_pneumonia=False, size=(128, 128), seed=None):
    """
    Renders an anatomically realistic chest X-ray radiograph image.
    Normal: Radiopaque thoracic cage, central cardiac silhouette, dark radiolucent clear lung fields.
    Pneumonia: Confluent alveolar consolidation, patchy opacification, air bronchograms, and ground-glass haziness.
    """
    if seed is not None:
        np.random.seed(seed)
        
    width, height = size
    # Dark thoracic background
    base = np.zeros((height, width), dtype=np.float32)
    
    # 1. Soft tissue thoracic contour & rib cage symmetry
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Outer chest boundary mask
    chest_ell = ((xx / 0.88) ** 2 + ((yy - 0.05) / 0.95) ** 2)
    chest_mask = np.clip(1.0 - chest_ell, 0.0, 1.0)
    base += chest_mask * 40.0
    
    # 2. Left and Right Lung Cavities (radiolucent = darker on X-ray)
    left_lung = np.exp(-(((xx + 0.42) / 0.28) ** 2 + ((yy + 0.05) / 0.65) ** 2) * 3.5)
    right_lung = np.exp(-(((xx - 0.42) / 0.28) ** 2 + ((yy + 0.05) / 0.65) ** 2) * 3.5)
    lung_fields = (left_lung + right_lung) * 85.0
    
    # Lungs are air-filled (darker in radiograph, so subtract from base soft tissue)
    base = np.clip(base - lung_fields, 10.0, 255.0)
    
    # 3. Mediastinum & Cardiac Silhouette (white/radiopaque)
    # Heart tilts toward anatomical left (viewer's right, positive xx)
    cardiac = np.exp(-(((xx - 0.12) / 0.32) ** 2 + ((yy - 0.25) / 0.38) ** 2) * 4.0) * 110.0
    mediastinum = np.exp(-((xx / 0.14) ** 2) * 12.0) * (1.0 - np.clip(yy * 0.4, -0.3, 0.8)) * 95.0
    base += cardiac + mediastinum
    
    # 4. Posterior & Anterior Ribs (subtle curvilinear striations)
    for rib_idx in range(-3, 4):
        rib_y = rib_idx * 0.22
        rib_band = np.exp(-((yy - rib_y - 0.08 * (xx ** 2)) / 0.035) ** 2 * 8.0) * 22.0
        base += rib_band
        
    # 5. Bronchovascular markings (fine branched arborization)
    noise = np.random.normal(0, 7.0, (height, width))
    base += noise
    
    # 6. Pneumonia Pathology: Infiltrates / Consolidations
    if is_pneumonia:
        # Patchy alveolar consolidation in middle or lower lobe
        side = np.random.choice([-1, 1])
        patch_x = side * np.random.uniform(0.28, 0.48)
        patch_y = np.random.uniform(-0.1, 0.4)
        patch_radius = np.random.uniform(0.22, 0.35)
        
        infiltrate_1 = np.exp(-(((xx - patch_x) / patch_radius) ** 2 + ((yy - patch_y) / (patch_radius * 1.2)) ** 2) * 4.5) * np.random.uniform(70.0, 110.0)
        # Secondary patchy ground glass opacity
        patch_x2 = -side * np.random.uniform(0.22, 0.40)
        patch_y2 = np.random.uniform(0.0, 0.35)
        infiltrate_2 = np.exp(-(((xx - patch_x2) / 0.25) ** 2 + ((yy - patch_y2) / 0.28) ** 2) * 5.0) * np.random.uniform(40.0, 80.0)
        
        # Micro-textures inside infiltrate
        coarse_noise = np.random.normal(15, 12, (height, width)) * ((infiltrate_1 + infiltrate_2) / 100.0)
        base += infiltrate_1 + infiltrate_2 + coarse_noise

    # Normalize to 0-255 uint8 grayscale then convert to RGB
    base = np.clip(base, 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img_rgb = Image.merge("RGB", (img, img, img))
    return img_rgb


def generate_image_dataset(n_train_per_class=60, n_test_per_class=20):
    """
    Generates training and testing chest radiograph datasets.
    """
    splits = {
        'train': {
            'normal': n_train_per_class,
            'pneumonia': n_train_per_class
        },
        'test': {
            'normal': n_test_per_class,
            'pneumonia': n_test_per_class
        }
    }
    
    seed = 100
    for split_name, class_dict in splits.items():
        for class_label, count in class_dict.items():
            target_dir = os.path.join(IMAGES_DIR, split_name, class_label)
            os.makedirs(target_dir, exist_ok=True)
            is_pneu = (class_label == 'pneumonia')
            
            for i in range(count):
                seed += 1
                img = draw_chest_radiograph(is_pneumonia=is_pneu, size=(64, 64), seed=seed)
                file_path = os.path.join(target_dir, f"{class_label}_{i+1:03d}.png")
                img.save(file_path)
                
    # Also save larger sample images for live UI demonstration
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    sample_normal = draw_chest_radiograph(is_pneumonia=False, size=(256, 256), seed=999)
    sample_normal.save(os.path.join(SAMPLES_DIR, "sample_normal_chest_xray.png"))
    
    sample_pneu = draw_chest_radiograph(is_pneumonia=True, size=(256, 256), seed=888)
    sample_pneu.save(os.path.join(SAMPLES_DIR, "sample_pneumonia_chest_xray.png"))
    
    print(f"Generated {n_train_per_class*2} training images and {n_test_per_class*2} test images.")
    print(f"Sample demonstration radiographs saved in {SAMPLES_DIR}")


if __name__ == '__main__':
    generate_tabular_data()
    generate_image_dataset()
    print("All datasets generated successfully!")
