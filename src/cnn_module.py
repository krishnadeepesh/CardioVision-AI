"""
Module 4: Convolutional Neural Network (CNN) Image Classification Pipeline
Domain: Healthcare (Chest Radiograph Medical Diagnostics)
Task: Classify Chest X-Ray Images as 'Normal' or 'Pneumonia / Pathological'
Framework: Keras 3 with Torch Backend
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Ensure Keras uses torch backend and caches to D: drive
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".keras"))

import keras
from keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "images")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")

CLASS_NAMES = ["Normal", "Pneumonia"]
IMG_SIZE = (64, 64)


def load_dataset(split="train"):
    """
    Loads images and labels from data/images/{split}/normal and data/images/{split}/pneumonia.
    Returns normalized numpy arrays (X, y_categorical).
    """
    split_dir = os.path.join(DATA_DIR, split)
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(["normal", "pneumonia"]):
        class_folder = os.path.join(split_dir, class_name)
        pattern = os.path.join(class_folder, "*.png")
        file_paths = glob.glob(pattern)
        
        for fp in file_paths:
            img = Image.open(fp).convert("RGB").resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(class_idx)
            
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    # One-hot encode targets for categorical cross-entropy
    y_cat = keras.utils.to_categorical(y, num_classes=2)
    return X, y, y_cat


def build_cnn_model():
    """
    Constructs a lightweight Convolutional Neural Network architecture.
    Layers:
    1. Input layer (64, 64, 3)
    2. Conv2D (32 filters, 3x3 kernel, ReLU) + BatchNormalization + MaxPooling2D (2x2)
    3. Conv2D (64 filters, 3x3 kernel, ReLU) + BatchNormalization + MaxPooling2D (2x2)
    4. Conv2D (128 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
    5. Flatten
    6. Dense (64 units, ReLU) + Dropout (0.3)
    7. Output Dense (2 units, Softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate_cnn(epochs=12, batch_size=16):
    """
    Trains the Keras CNN on training set and evaluates on independent test set.
    """
    print("Loading Chest Radiograph Image Datasets...")
    X_train, y_train_labels, y_train = load_dataset("train")
    X_test, y_test_labels, y_test = load_dataset("test")
    
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    model = build_cnn_model()
    model.summary()
    
    print("\nTraining Keras CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    report = classification_report(y_test_labels, y_pred_labels, target_names=CLASS_NAMES)
    
    print("\n" + "="*50)
    print("MODULE 4 KERAS CNN TEST RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")
    print("\nClassification Report:\n", report)
    
    # Plot curves and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_training_history(history, test_acc, cm)
    
    # Save trained model weights
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, "cnn_chest_xray.keras")
    model.save(model_save_path)
    print(f"Saved trained CNN model to {model_save_path}")
    
    # Generate output evidence on a sample image for Deliverable 4
    demonstrate_sample_prediction(model)
    
    return {
        "model": model,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "confusion_matrix": cm,
        "report": report,
        "history": history.history
    }


def plot_training_history(history, test_acc, cm):
    """
    Generates training curves and confusion matrix plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Accuracy & Loss Curves
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    axes[0].plot(epochs_range, history.history['accuracy'], 'b-o', label='Training Accuracy')
    axes[0].plot(epochs_range, history.history['val_accuracy'], 'g-s', label='Validation Accuracy')
    axes[0].set_title(f'CNN Accuracy Progression (Final Test Acc: {test_acc*100:.2f}%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Confusion Matrix Heatmap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Purples',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1]
    )
    axes[1].set_title('CNN Confusion Matrix on Test Radiographs', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "cnn_training_performance.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved CNN performance curves to {curve_path}")


def demonstrate_sample_prediction(model=None):
    """
    Runs inference on demonstration sample images and generates visual output evidence.
    """
    if model is None:
        model_save_path = os.path.join(MODEL_DIR, "cnn_chest_xray.keras")
        model = keras.models.load_model(model_save_path)
        
    sample_files = [
        ("Normal Sample Radiograph", os.path.join(SAMPLES_DIR, "sample_normal_chest_xray.png")),
        ("Pathological / Pneumonia Sample", os.path.join(SAMPLES_DIR, "sample_pneumonia_chest_xray.png"))
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for idx, (title, fpath) in enumerate(sample_files):
        if not os.path.exists(fpath):
            continue
        img_pil = Image.open(fpath).convert("RGB")
        img_resized = img_pil.resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)
        
        preds = model.predict(arr, verbose=0)[0]
        pred_class = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds) * 100.0
        
        axes[idx].imshow(img_pil)
        axes[idx].axis("off")
        color = "darkgreen" if pred_class == "Normal" else "darkred"
        axes[idx].set_title(
            f"{title}\nPredicted: {pred_class} ({confidence:.2f}%)",
            fontsize=11, fontweight="bold", color=color
        )
        
    plt.tight_layout()
    evidence_path = os.path.join(OUTPUT_DIR, "cnn_prediction_evidence.png")
    plt.savefig(evidence_path, dpi=200)
    plt.close()
    print(f"Saved sample prediction evidence to {evidence_path}")


def predict_single_image(image_input, model=None):
    """
    Inference helper for Streamlit UI.
    Takes PIL Image or image file path, returns predicted label, confidence, and class probabilities.
    """
    if model is None:
        model_save_path = os.path.join(MODEL_DIR, "cnn_chest_xray.keras")
        model = keras.models.load_model(model_save_path)
        
    if isinstance(image_input, str):
        img_pil = Image.open(image_input).convert("RGB")
    else:
        img_pil = image_input.convert("RGB")
        
    img_resized = img_pil.resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)
    
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[0]: float(probs[0]),
            CLASS_NAMES[1]: float(probs[1])
        },
        "is_pathological": (pred_idx == 1)
    }


if __name__ == '__main__':
    train_and_evaluate_cnn()
