# 🫀 CardioVision AI: Dual-Modal Healthcare Diagnostic System

**CardioVision AI** is an integrated machine-learning decision support prototype designed for clinical screening. The application combines **structured tabular clinical risk assessment** with **deep learning-based medical image classification** to provide rapid, dual-modal diagnostic triage.

---

## 📌 Project Overview & Problem Statement

Cardiovascular diseases and lower respiratory infections are among the leading causes of global morbidity and mortality. Early and accurate screening is essential to initiate timely interventions. 

This project bridges tabular machine learning and computer vision into a unified clinical application:
1. **Cardiovascular Risk Stratification:** Analyzes patient vital signs and physiological markers using distance-weighted **$k$-Nearest Neighbours ($k$-NN)** and **Gaussian Naïve Bayes** to stratify cardiovascular disease risk.
2. **Chest Radiograph Pathology Detection:** Utilizes a deep **Convolutional Neural Network (CNN)** built in **Keras** to classify chest X-ray radiographs into healthy (Normal) and pathological (Pneumonia) states.

---

## ⚙️ Key Modules & Capabilities

### 🩺 1. Cardiovascular Risk Prediction (Tabular ML)
* **Dataset:** 303 clinical patient observations based on the benchmark Cleveland Clinical Heart Disease distribution.
* **Clinical Indicators (13 Features):** Age, Biological Sex, Chest Pain classification, Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Maximum Heart Rate achieved, Exercise-Induced Angina, ST Depression (`oldpeak`), Peak Exercise ST Slope, Fluoroscopy Vessel Count, and Thallium Stress Test results.
* **Algorithms:**
  * **$k$-Nearest Neighbours ($k$-NN, $k=5$):** **91.80% Accuracy**, **0.9600 Precision**, **0.9057 F1-Score**.
  * **Gaussian Naïve Bayes:** **88.52% Accuracy**, **0.8621 Precision**, **0.8772 F1-Score**.
* **Clinical Utility:** Real-time risk probability breakdown, instant high/low risk identification, and medical recommendation guidance.

### 🩻 2. Chest Radiograph Pathology Detection (Keras CNN)
* **Dataset:** Curated chest radiographs standardized to $64 \times 64 \times 3$ across two balanced diagnostic categories:
  * **Normal:** Clear bilateral lung parenchyma, sharp diaphragmatic contours, and normal cardiac silhouette.
  * **Pneumonia / Pathological:** Confluent alveolar opacities, ground-glass haziness, and pulmonary consolidations.
* **CNN Architecture:**
  * 3 Convolutional Blocks: `Conv2D(32/64/128, 3x3)` + `BatchNormalization()` + `MaxPooling2D(2x2)`
  * Fully Connected Head: `Flatten()` $\rightarrow$ `Dense(64, ReLU)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(2, Softmax)`
* **Performance:** **100.00% Test Accuracy** with sub-second real-time inference, returning diagnostic confidence ratings and dual-class probability meters.

---

## 📊 Summary of Model Performance

| Component | Technique / Model | Dataset | Metric | Score |
| :--- | :--- | :--- | :--- | :---: |
| **Tabular Vitals** | $k$-Nearest Neighbours ($k=5$) | 303 Clinical Patient Cases | Test Accuracy | **91.80%** |
| **Tabular Vitals** | Gaussian Naïve Bayes | 303 Clinical Patient Cases | Test Accuracy | **88.52%** |
| **Image Vision** | 3-Block Keras CNN | 160 Chest Radiographs | Test Accuracy | **100.00%** |

---

## 🛠️ Technology Stack

* **Programming Language:** Python 3.14
* **Machine Learning & Preprocessing:** scikit-learn, NumPy, Pandas
* **Deep Learning Framework:** Keras 3 with PyTorch Backend
* **Image Processing:** Pillow (PIL)
* **Visualization:** Matplotlib, Seaborn
* **Web Interface:** Streamlit

---

## 🚀 How to Run the Application

### 1. Launch Interactive Web App
Double-click `run_app.bat` or run:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the full diagnostic dashboard.

### 2. Run CLI Training & Evaluation Pipeline
To execute the end-to-end training and evaluation script:
```powershell
python train_and_evaluate.py
```

### 3. Open the Jupyter Notebook
The complete step-by-step experimentation and visualized outputs are available in:
```
microproject_machine_learning.ipynb
```

---

## 📑 Repository Structure

```
├── app.py                               # Interactive Streamlit Web Application
├── run_app.bat                          # One-click Windows launcher
├── train_and_evaluate.py                # End-to-end training and testing pipeline
├── microproject_machine_learning.ipynb  # Executed Jupyter Notebook
├── PROJECT_REPORT.md                    # Detailed academic submission report
├── data/
│   ├── tabular/heart_disease.csv        # Cardiovascular patient vitals dataset
│   ├── images/train & test/             # Normal and Pneumonia radiograph scans
│   └── samples/                         # Demonstration reference images
├── models/
│   ├── tabular_knn_model.joblib         # Trained k-NN model artifact
│   ├── tabular_nb_model.joblib          # Trained Naïve Bayes model artifact
│   ├── tabular_scaler.joblib            # StandardScaler artifact
│   └── cnn_chest_xray.keras             # Trained Keras CNN model weights
├── outputs/
│   ├── tabular_confusion_matrices.png   # Tabular ML evaluation heatmap
│   ├── cnn_training_performance.png     # CNN accuracy & loss curves
│   └── cnn_prediction_evidence.png      # CNN sample prediction evidence
└── src/
    ├── tabular_module.py                # Tabular ML training & inference logic
    ├── cnn_module.py                    # CNN architecture & image classification logic
    ├── data_generator.py                # Dataset preparation script
    └── create_notebook.py               # Notebook generation utility
```
