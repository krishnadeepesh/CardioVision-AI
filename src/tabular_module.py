"""
Module 2: Tabular Machine Learning Pipeline
Domain: Healthcare (Cardiovascular Risk Stratification)
Algorithms: k-Nearest Neighbours (k-NN) & Gaussian Naïve Bayes
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tabular", "heart_disease.csv"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))


def load_and_preprocess_data(csv_path=DATA_PATH, test_size=0.2, random_state=42):
    """
    Loads clinical cardiovascular dataset and splits into train/test sets with scaling.
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'df': df,
        'feature_names': feature_names,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }


def train_knn_model(data_dict, n_neighbors=5):
    """
    Trains a k-Nearest Neighbour classifier.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    y_pred = knn.predict(data_dict['X_test_scaled'])
    y_prob = knn.predict_proba(data_dict['X_test_scaled'])[:, 1]
    
    acc = accuracy_score(data_dict['y_test'], y_pred)
    prec = precision_score(data_dict['y_test'], y_pred)
    rec = recall_score(data_dict['y_test'], y_pred)
    f1 = f1_score(data_dict['y_test'], y_pred)
    cm = confusion_matrix(data_dict['y_test'], y_pred)
    report = classification_report(data_dict['y_test'], y_pred, target_names=['Low Risk (0)', 'High Risk (1)'])
    
    return {
        'model': knn,
        'algorithm': f'k-Nearest Neighbours (k={n_neighbors})',
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'report': report
    }


def train_naive_bayes_model(data_dict):
    """
    Trains a Gaussian Naïve Bayes classifier.
    """
    gnb = GaussianNB()
    gnb.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    y_pred = gnb.predict(data_dict['X_test_scaled'])
    y_prob = gnb.predict_proba(data_dict['X_test_scaled'])[:, 1]
    
    acc = accuracy_score(data_dict['y_test'], y_pred)
    prec = precision_score(data_dict['y_test'], y_pred)
    rec = recall_score(data_dict['y_test'], y_pred)
    f1 = f1_score(data_dict['y_test'], y_pred)
    cm = confusion_matrix(data_dict['y_test'], y_pred)
    report = classification_report(data_dict['y_test'], y_pred, target_names=['Low Risk (0)', 'High Risk (1)'])
    
    return {
        'model': gnb,
        'algorithm': 'Gaussian Naïve Bayes',
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'report': report
    }


def plot_tabular_results(knn_results, nb_results, data_dict):
    """
    Plots confusion matrices and comparative metrics.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Confusion Matrix for k-NN
    sns.heatmap(
        knn_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
        xticklabels=['Low Risk (0)', 'High Risk (1)'],
        yticklabels=['Low Risk (0)', 'High Risk (1)'],
        ax=axes[0]
    )
    axes[0].set_title(f"Confusion Matrix: {knn_results['algorithm']}\nAccuracy: {knn_results['accuracy']*100:.2f}%", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Diagnostic Label')
    axes[0].set_ylabel('True Clinical Label')
    
    # 2. Confusion Matrix for Naive Bayes
    sns.heatmap(
        nb_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
        xticklabels=['Low Risk (0)', 'High Risk (1)'],
        yticklabels=['Low Risk (0)', 'High Risk (1)'],
        ax=axes[1]
    )
    axes[1].set_title(f"Confusion Matrix: {nb_results['algorithm']}\nAccuracy: {nb_results['accuracy']*100:.2f}%", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Diagnostic Label')
    axes[1].set_ylabel('True Clinical Label')
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "tabular_confusion_matrices.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved tabular evaluation plot to {plot_path}")
    return plot_path


def save_tabular_artifacts(data_dict, knn_results, nb_results):
    """
    Saves trained models and preprocessing scaler for deployment.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    scaler_path = os.path.join(MODEL_DIR, "tabular_scaler.joblib")
    knn_path = os.path.join(MODEL_DIR, "tabular_knn_model.joblib")
    nb_path = os.path.join(MODEL_DIR, "tabular_nb_model.joblib")
    
    joblib.dump(data_dict['scaler'], scaler_path)
    joblib.dump(knn_results['model'], knn_path)
    joblib.dump(nb_results['model'], nb_path)
    print("Saved tabular model artifacts to models/")


def predict_patient_risk(patient_dict, model_choice='knn'):
    """
    Inference helper for real-time clinical patient risk prediction.
    """
    scaler_path = os.path.join(MODEL_DIR, "tabular_scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    if model_choice == 'knn':
        model = joblib.load(os.path.join(MODEL_DIR, "tabular_knn_model.joblib"))
    else:
        model = joblib.load(os.path.join(MODEL_DIR, "tabular_nb_model.joblib"))
        
    df_patient = pd.DataFrame([patient_dict])
    scaled_feats = scaler.transform(df_patient)
    pred_class = int(model.predict(scaled_feats)[0])
    pred_probs = model.predict_proba(scaled_feats)[0]
    
    risk_label = "High Risk / Disease Indicated" if pred_class == 1 else "Low Risk / No Disease Detected"
    return {
        'prediction': pred_class,
        'risk_label': risk_label,
        'probability_low_risk': float(pred_probs[0]),
        'probability_high_risk': float(pred_probs[1]),
        'confidence': float(max(pred_probs))
    }


if __name__ == '__main__':
    data = load_and_preprocess_data()
    knn_res = train_knn_model(data, n_neighbors=5)
    nb_res = train_naive_bayes_model(data)
    
    print("\n" + "="*50)
    print("MODULE 2 TABULAR ML RESULTS (HEALTHCARE)")
    print("="*50)
    print(f"Algorithm 1: {knn_res['algorithm']}")
    print(f"Accuracy: {knn_res['accuracy']*100:.2f}% | F1-Score: {knn_res['f1']:.4f}")
    print("\nClassification Report:\n", knn_res['report'])
    
    print(f"Algorithm 2: {nb_res['algorithm']}")
    print(f"Accuracy: {nb_res['accuracy']*100:.2f}% | F1-Score: {nb_res['f1']:.4f}")
    print("\nClassification Report:\n", nb_res['report'])
    
    plot_tabular_results(knn_res, nb_res, data)
    save_tabular_artifacts(data, knn_res, nb_res)
