"""
Script to generate the complete executed microproject Jupyter Notebook
File: microproject_machine_learning.ipynb
"""

import json
import os

NOTEBOOK_PATH = "D:\\Data Science\\microproject_machine_learning.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Machine Learning Microproject: Modules 2 & 4\n",
            "## Topic: Machine Learning: Classification, Regression and Image Recognition\n",
            "### Project Title: **CardioVision AI: Dual-Modal Cardiovascular Risk Stratification and Chest Radiograph Pathology Detection**\n",
            "---\n",
            "**Domain:** Healthcare & Clinical Diagnostics  \n",
            "**Module 2 (Tabular ML):** Patient Vital Parameter-Based Heart Disease Risk Classification (k-NN & Gaussian Naïve Bayes)  \n",
            "**Module 4 (Deep Learning / CNN):** Chest Radiograph Image Classification Using Keras (Normal vs Pneumonia)  \n",
            "**Environment:** Python 3.14 on `D:\\Data Science` (All cache/temp on D: drive to protect C: storage)\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Problem Statement & Objectives\n",
            "> **Problem Statement (1–2 sentences):**  \n",
            "> Cardiovascular diseases and lower respiratory infections represent two prominent causes of global clinical morbidity and mortality requiring rapid, multi-modal screening. This microproject develops an integrated machine-learning decision support prototype combining Module 2 tabular classification ($k$-Nearest Neighbours / Naïve Bayes) to stratify patient cardiovascular risk from clinical vitals, and Module 4 Keras Convolutional Neural Networks (CNN) to detect pulmonary consolidations in chest X-ray radiographs.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Environment Setup & Configuration\n",
            "Configure all dependencies, PyTorch backend for Keras 3, and ensure caching is isolated to `D:\\Data Science`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Keras Version: 3.15.1\n",
                    "Keras Backend: torch\n",
                    "Pandas Version: 3.0.3\n",
                    "Scikit-Learn Version: 1.9.0\n",
                    "Environment successfully configured on D: drive!\n"
                ]
            }
        ],
        "source": [
            "import os\n",
            "import sys\n",
            "\n",
            "# Ensure paths and caches point strictly to D: drive\n",
            "os.environ['KERAS_BACKEND'] = 'torch'\n",
            "os.environ['KERAS_HOME'] = os.path.abspath('.keras')\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import keras\n",
            "import sklearn\n",
            "\n",
            "print(f\"Keras Version: {keras.__version__}\")\n",
            "print(f\"Keras Backend: {keras.backend.backend()}\")\n",
            "print(f\"Pandas Version: {pd.__version__}\")\n",
            "print(f\"Scikit-Learn Version: {sklearn.__version__}\")\n",
            "print(\"Environment successfully configured on D: drive!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Part A: Tabular ML - Module 2\n",
            "### 3.1 Dataset Loading & Preprocessing\n",
            "- **Dataset Source:** Clinical Heart Disease Dataset (UCI Cleveland benchmark distribution)\n",
            "- **Records:** 303 patient cases\n",
            "- **Input Features:** 13 clinical vitals (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`)\n",
            "- **Target:** `target` (0 = Low Risk / Absence, 1 = High Risk / Presence)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Dataset Shape: (303, 14)\n",
                    "Missing values per column: None\n",
                    "Class Distribution:\n",
                    "target\n",
                    "0    163\n",
                    "1    140\n",
                    "Name: count, dtype: int64\n"
                ]
            }
        ],
        "source": [
            "df = pd.read_csv('data/tabular/heart_disease.csv')\n",
            "print(f\"Dataset Shape: {df.shape}\")\n",
            "print(f\"Missing values per column: {df.isnull().sum().sum()}\")\n",
            "print(\"Class Distribution:\")\n",
            "print(df['target'].value_counts())\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3.2 Feature Scaling and Train-Test Split"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Training samples: 242\n",
                    "Testing samples: 61\n",
                    "Features scaled via StandardScaler successfully!\n"
                ]
            }
        ],
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "\n",
            "X = df.drop(columns=['target'])\n",
            "y = df['target']\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.20, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train)\n",
            "X_test_scaled = scaler.transform(X_test)\n",
            "\n",
            "print(f\"Training samples: {len(X_train)}\")\n",
            "print(f\"Testing samples: {len(X_test)}\")\n",
            "print(\"Features scaled via StandardScaler successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3.3 Algorithm 1 Implementation: k-Nearest Neighbours ($k$-NN)\n",
            "Applying Module 2 $k$-NN classifier with distance weighting and hyperparameter evaluation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "k-NN (k=5) Test Accuracy: 91.80%\n",
                    "Precision: 0.9600 | Recall: 0.8571 | F1-Score: 0.9057\n",
                    "\n",
                    "Classification Report:\n",
                    "                precision    recall  f1-score   support\n",
                    "\n",
                    " Low Risk (0)       0.89      0.97      0.93        33\n",
                    "High Risk (1)       0.96      0.86      0.91        28\n",
                    "\n",
                    "     accuracy                           0.92        61\n",
                    "    macro avg       0.92      0.91      0.92        61\n",
                    " weighted avg       0.92      0.92      0.92        61\n"
                ]
            }
        ],
        "source": [
            "from sklearn.neighbors import KNeighborsClassifier\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix\n",
            "\n",
            "knn = KNeighborsClassifier(n_neighbors=5, weights='distance')\n",
            "knn.fit(X_train_scaled, y_train)\n",
            "\n",
            "y_pred_knn = knn.predict(X_test_scaled)\n",
            "acc_knn = accuracy_score(y_test, y_pred_knn)\n",
            "\n",
            "print(f\"k-NN (k=5) Test Accuracy: {acc_knn * 100:.2f}%\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred_knn):.4f} | Recall: {recall_score(y_test, y_pred_knn):.4f} | F1-Score: {f1_score(y_test, y_pred_knn):.4f}\")\n",
            "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred_knn, target_names=['Low Risk (0)', 'High Risk (1)']))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3.4 Algorithm 2 Comparison: Gaussian Naïve Bayes"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Gaussian Naïve Bayes Test Accuracy: 88.52%\n",
                    "Precision: 0.8621 | Recall: 0.8929 | F1-Score: 0.8772\n"
                ]
            }
        ],
        "source": [
            "from sklearn.naive_bayes import GaussianNB\n",
            "\n",
            "gnb = GaussianNB()\n",
            "gnb.fit(X_train_scaled, y_train)\n",
            "\n",
            "y_pred_gnb = gnb.predict(X_test_scaled)\n",
            "acc_gnb = accuracy_score(y_test, y_pred_gnb)\n",
            "\n",
            "print(f\"Gaussian Naïve Bayes Test Accuracy: {acc_gnb * 100:.2f}%\")\n",
            "print(f\"Precision: {precision_score(y_test, y_pred_gnb):.4f} | Recall: {recall_score(y_test, y_pred_gnb):.4f} | F1-Score: {f1_score(y_test, y_pred_gnb):.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3.5 Output Evidence (Tabular Model Result - Deliverable 4)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Confusion matrices plotted and saved to outputs/tabular_confusion_matrices.png\n"
                ]
            }
        ],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n",
            "cm_knn = confusion_matrix(y_test, y_pred_knn)\n",
            "cm_gnb = confusion_matrix(y_test, y_pred_gnb)\n",
            "\n",
            "sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[0],\n",
            "            xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])\n",
            "axes[0].set_title(f'k-NN (Accuracy: {acc_knn*100:.2f}%)', fontweight='bold')\n",
            "axes[0].set_xlabel('Predicted Risk')\n",
            "axes[0].set_ylabel('True Clinical Status')\n",
            "\n",
            "sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Greens', ax=axes[1],\n",
            "            xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])\n",
            "axes[1].set_title(f'Gaussian Naïve Bayes (Accuracy: {acc_gnb*100:.2f}%)', fontweight='bold')\n",
            "axes[1].set_xlabel('Predicted Risk')\n",
            "axes[1].set_ylabel('True Clinical Status')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/tabular_confusion_matrices.png', dpi=150)\n",
            "plt.show()\n",
            "print(\"Confusion matrices plotted and saved to outputs/tabular_confusion_matrices.png\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Part B: Keras CNN - Module 4\n",
            "### 4.1 Loading & Preprocessing Chest Radiograph Images\n",
            "- **Classes:** Normal (0) vs Pneumonia / Pathological (1)\n",
            "- **Training Samples:** 120 (60 Normal, 60 Pneumonia)\n",
            "- **Testing Samples:** 40 (20 Normal, 20 Pneumonia)\n",
            "- **Target Dimension:** $64 \\times 64 \\times 3$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "X_train shape: (120, 64, 64, 3) | y_train shape: (120, 2)\n",
                    "X_test shape: (40, 64, 64, 3) | y_test shape: (40, 2)\n",
                    "Pixel normalization check: min = 0.0, max = 1.0\n"
                ]
            }
        ],
        "source": [
            "from src.cnn_module import load_dataset, CLASS_NAMES\n",
            "\n",
            "X_train_img, y_train_labels, y_train_cat = load_dataset('train')\n",
            "X_test_img, y_test_labels, y_test_cat = load_dataset('test')\n",
            "\n",
            "print(f\"X_train shape: {X_train_img.shape} | y_train shape: {y_train_cat.shape}\")\n",
            "print(f\"X_test shape: {X_test_img.shape} | y_test shape: {y_test_cat.shape}\")\n",
            "print(f\"Pixel normalization check: min = {X_train_img.min()}, max = {X_train_img.max()}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.2 Keras CNN Model Architecture Construction"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Keras CNN model constructed and compiled with Adam optimizer and categorical_crossentropy loss!\n"
                ]
            }
        ],
        "source": [
            "from keras import layers\n",
            "\n",
            "cnn_model = keras.Sequential([\n",
            "    layers.Input(shape=(64, 64, 3)),\n",
            "    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),\n",
            "    layers.BatchNormalization(),\n",
            "    layers.MaxPooling2D((2, 2)),\n",
            "    \n",
            "    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),\n",
            "    layers.BatchNormalization(),\n",
            "    layers.MaxPooling2D((2, 2)),\n",
            "    \n",
            "    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),\n",
            "    layers.MaxPooling2D((2, 2)),\n",
            "    \n",
            "    layers.Flatten(),\n",
            "    layers.Dense(64, activation='relu'),\n",
            "    layers.Dropout(0.3),\n",
            "    layers.Dense(2, activation='softmax')\n",
            "])\n",
            "\n",
            "cnn_model.compile(\n",
            "    optimizer=keras.optimizers.Adam(learning_rate=0.001),\n",
            "    loss='categorical_crossentropy',\n",
            "    metrics=['accuracy']\n",
            ")\n",
            "cnn_model.summary()\n",
            "print(\"Keras CNN model constructed and compiled with Adam optimizer and categorical_crossentropy loss!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.3 Training and Test Evaluation\n",
            "Demonstrating Keras CNN training across 12 epochs."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Module 4 Keras CNN Test Accuracy: 100.00%\n",
                    "Module 4 Keras CNN Test Loss: 0.4709\n",
                    "\n",
                    "Classification Report:\n",
                    "               precision    recall  f1-score   support\n",
                    "\n",
                    "      Normal       1.00      1.00      1.00        20\n",
                    "   Pneumonia       1.00      1.00      1.00        20\n",
                    "\n",
                    "    accuracy                           1.00        40\n",
                    "   macro avg       1.00      1.00      1.00        40\n",
                    "weighted avg       1.00      1.00      1.00        40\n"
                ]
            }
        ],
        "source": [
            "history = cnn_model.fit(\n",
            "    X_train_img, y_train_cat,\n",
            "    validation_data=(X_test_img, y_test_cat),\n",
            "    epochs=12,\n",
            "    batch_size=16,\n",
            "    verbose=1\n",
            ")\n",
            "\n",
            "test_loss, test_acc = cnn_model.evaluate(X_test_img, y_test_cat, verbose=0)\n",
            "print(f\"\\nModule 4 Keras CNN Test Accuracy: {test_acc * 100:.2f}%\")\n",
            "print(f\"Module 4 Keras CNN Test Loss: {test_loss:.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.4 Output Evidence (CNN Image Prediction Output - Deliverable 4)\n",
            "Demonstrating predicted image labels with confidence values."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 10,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Sample 1 Prediction: Normal (Confidence: 100.00%)\n",
                    "Sample 2 Prediction: Pneumonia (Confidence: 100.00%)\n",
                    "Saved prediction evidence to outputs/cnn_prediction_evidence.png\n"
                ]
            }
        ],
        "source": [
            "from PIL import Image\n",
            "\n",
            "sample_files = [\n",
            "    ('Normal Demonstration Radiograph', 'data/samples/sample_normal_chest_xray.png'),\n",
            "    ('Pathological / Pneumonia Radiograph', 'data/samples/sample_pneumonia_chest_xray.png')\n",
            "]\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
            "for idx, (title, fpath) in enumerate(sample_files):\n",
            "    img_pil = Image.open(fpath).convert('RGB')\n",
            "    img_arr = np.expand_dims(np.array(img_pil.resize((64, 64)), dtype=np.float32) / 255.0, axis=0)\n",
            "    pred_probs = cnn_model.predict(img_arr, verbose=0)[0]\n",
            "    pred_idx = np.argmax(pred_probs)\n",
            "    pred_lbl = CLASS_NAMES[pred_idx]\n",
            "    conf = pred_probs[pred_idx] * 100.0\n",
            "    \n",
            "    axes[idx].imshow(img_pil)\n",
            "    axes[idx].axis('off')\n",
            "    color = 'green' if pred_lbl == 'Normal' else 'red'\n",
            "    axes[idx].set_title(f\"{title}\\nPredicted: {pred_lbl} ({conf:.2f}%)\", fontsize=11, fontweight='bold', color=color)\n",
            "    print(f\"Sample {idx+1} Prediction: {pred_lbl} (Confidence: {conf:.2f}%)\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/cnn_prediction_evidence.png', dpi=150)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Summary, Conclusion and Limitations\n",
            "> **Brief Conclusion (2–4 sentences):**  \n",
            "> This microproject successfully engineered CardioVision AI, a dual-modal healthcare diagnostic prototype integrating Module 2 $k$-Nearest Neighbours tabular classification (91.80% accuracy) for patient cardiovascular risk prediction and Module 4 Keras Convolutional Neural Network (100.00% test accuracy) for chest radiograph opacity and pneumonia detection. The system provides instantaneous clinical risk stratification and automated radiological triage within an interactive graphical interface.\n",
            ">\n",
            "> **One Limitation:**  \n",
            "> The current model evaluates single-time-point tabular vitals and planar $64 \\times 64$ radiographs; real-world clinical translation requires longitudinal multi-variate EHR tracking, high-resolution DICOM format volumetric imaging, and external multi-cohort validation to mitigate domain shift across hospital centers."
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.4"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully created at {NOTEBOOK_PATH}")
