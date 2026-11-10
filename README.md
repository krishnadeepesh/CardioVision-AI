# CardioVision AI: Machine Learning Microproject

An integrated machine-learning application for **Cardiovascular Disease Risk Stratification** and **Chest Radiograph Image Recognition** built with Python, scikit-learn, Keras 3, PyTorch, and Streamlit.

---

## 🚀 Quick Launch Instructions

### Option 1: One-Click GUI Launcher (Windows)
Double-click `run_app.bat` inside `D:\Data Science`. This automatically launches the Streamlit web application at `http://localhost:8501`.

### Option 2: Command Line (PowerShell / Command Prompt)
Run the following commands:
```powershell
cd /d "D:\Data Science"
& "D:\Data Science\venv\Scripts\streamlit.exe" run app.py
```

### Option 3: End-to-End Retraining & Metric Regeneration
To retrain both models and regenerate all plots in `outputs/`:
```powershell
& "D:\Data Science\venv\Scripts\python.exe" train_and_evaluate.py
```

---

## 📁 Repository Structure
```
D:\Data Science\
├── .cache/                              # Local cache directory (D: drive)
├── .keras/                              # Keras configuration & weights (D: drive)
├── .pip_cache/                          # Isolated pip cache
├── .tmp/                                # Isolated temporary folder
├── data/
│   ├── tabular/
│   │   └── heart_disease.csv            # UCI Cleveland benchmark dataset (303 records)
│   ├── images/
│   │   ├── train/ (normal, pneumonia)   # 120 training radiographs
│   │   └── test/ (normal, pneumonia)    # 40 test radiographs
│   └── samples/                         # Demonstration radiographs for live testing
├── models/
│   ├── tabular_knn_model.joblib         # Trained k-NN model
│   ├── tabular_nb_model.joblib          # Trained Gaussian Naïve Bayes model
│   ├── tabular_scaler.joblib            # StandardScaler artifact
│   └── cnn_chest_xray.keras             # Trained Keras CNN model weights
├── outputs/
│   ├── tabular_confusion_matrices.png   # Module 2 evaluation plot
│   ├── cnn_training_performance.png     # Module 4 training curves & confusion matrix
│   └── cnn_prediction_evidence.png      # Module 4 prediction evidence
├── src/
│   ├── data_generator.py                # Dataset synthesizer & preprocessor
│   ├── tabular_module.py                # Module 2 algorithms (k-NN & Naïve Bayes)
│   ├── cnn_module.py                    # Module 4 Keras CNN pipeline
│   └── create_notebook.py               # Jupyter notebook generator
├── app.py                               # Interactive Streamlit Web Application
├── run_app.bat                          # One-click Windows batch launcher
├── train_and_evaluate.py                # Master CLI training & testing pipeline
├── microproject_machine_learning.ipynb  # Executed submission Jupyter Notebook
├── PROJECT_REPORT.md                    # Comprehensive 10/10 mark academic report
└── README.md                            # Quickstart guide
```

---


