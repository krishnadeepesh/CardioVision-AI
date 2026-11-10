"""
CardioVision AI - Modern Clinical Diagnostic Application
Modules:
- Cardiovascular Risk Assessment (k-NN & Naïve Bayes on Patient Vitals)
- Chest Radiograph Image Classification (Keras CNN for Pneumonia Detection)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Configure environment variables to stay on D: drive
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["PIP_CACHE_DIR"] = os.path.join(BASE_DIR, ".pip_cache")
os.environ["TEMP"] = os.path.join(BASE_DIR, ".tmp")
os.environ["TMP"] = os.path.join(BASE_DIR, ".tmp")
os.environ["KERAS_HOME"] = os.path.join(BASE_DIR, ".keras")
os.environ["TORCH_HOME"] = os.path.join(BASE_DIR, ".torch")
os.environ["KERAS_BACKEND"] = "torch"

from src.tabular_module import predict_patient_risk
from src.cnn_module import predict_single_image

# Page Setup
st.set_page_config(
    page_title="CardioVision AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1e293b;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 500;
        margin: 0;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #dcfce7;
        color: #15803d;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #bbf7d0;
    }
    
    /* Section Cards */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        height: 100%;
        margin-bottom: 12px;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Results Cards */
    .result-card-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1.5px solid #86efac;
        border-radius: 14px;
        padding: 24px;
        color: #14532d;
        box-shadow: 0 4px 6px -1px rgba(34, 197, 94, 0.1);
        margin-top: 20px;
    }
    .result-card-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1.5px solid #fca5a5;
        border-radius: 14px;
        padding: 24px;
        color: #7f1d1d;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.1);
        margin-top: 20px;
    }
    .res-badge {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-green { background: #bbf7d0; color: #166534; }
    .badge-red { background: #fecaca; color: #991b1b; }
    
    .big-score {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 8px 0;
    }

    /* Radiograph Preview Container */
    .radiograph-frame {
        background: #0f172a;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Modern Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.98rem;
        padding: 10px 22px;
        border-radius: 8px;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1e40af !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/caduceus.png", width=64)
    st.markdown("### **CardioVision AI**")
    st.caption("Intelligent Clinical Diagnostics")
    st.markdown("""
    <div class="status-badge">
        <span style="font-size: 10px;">●</span> System Online (D: Drive)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("##### 📌 **Diagnostic Engines**")
    st.markdown("""
    - **🫀 Vitals Engine:** $k$-NN & Naïve Bayes
    - **🩻 Vision Engine:** Multi-Layer Keras CNN
    """)
    st.markdown("---")
    st.markdown("##### 💾 **System Storage**")
    st.caption("Active Workspace: `D:\\Data Science`  \nZero `C:` drive consumption.")

# Main Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🫀 CardioVision AI</span>
    </div>
    <p class="hero-subtitle">
        Unified Clinical Decision Support for Cardiovascular Risk Stratification & Chest Radiograph Pathology Detection
    </p>
</div>
""", unsafe_allow_html=True)

# Top Prominent Navigation Tabs
tab_tabular, tab_image = st.tabs([
    "🩺 1. Cardiovascular Risk Assessment (Vitals)",
    "🩻 2. Chest Radiograph Diagnosis (Image CNN)"
])


# =====================================================================
# MODULE 1: HEART DISEASE RISK PREDICTION (TABULAR ML)
# =====================================================================
with tab_tabular:
    st.markdown("#### **Patient Health Vitals & Risk Analysis**")
    st.caption("Adjust clinical indicators below or choose a quick pre-configured test profile.")
    
    # Preset selection buttons
    p_c1, p_c2, p_c3 = st.columns([1, 1, 1])
    with p_c1:
        load_healthy = st.button("🟢 Load Normal / Low-Risk Profile", use_container_width=True)
    with p_c2:
        load_risk = st.button("🔴 Load High-Risk Profile", use_container_width=True)
    with p_c3:
        reset_vals = st.button("🔄 Reset Baseline Values", use_container_width=True)
        
    if "patient_vals" not in st.session_state:
        st.session_state.patient_vals = {
            "age": 52, "sex": 1, "cp": 1, "trestbps": 125, "chol": 215,
            "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 0.5,
            "slope": 0, "ca": 0, "thal": 2
        }
        
    if load_healthy:
        st.session_state.patient_vals = {
            "age": 42, "sex": 0, "cp": 2, "trestbps": 115, "chol": 185,
            "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.0,
            "slope": 0, "ca": 0, "thal": 2
        }
    elif load_risk:
        st.session_state.patient_vals = {
            "age": 63, "sex": 1, "cp": 0, "trestbps": 165, "chol": 290,
            "fbs": 1, "restecg": 1, "thalach": 120, "exang": 1, "oldpeak": 2.8,
            "slope": 1, "ca": 2, "thal": 3
        }
    elif reset_vals:
        st.session_state.patient_vals = {
            "age": 52, "sex": 1, "cp": 1, "trestbps": 125, "chol": 215,
            "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 0.5,
            "slope": 0, "ca": 0, "thal": 2
        }
        
    vals = st.session_state.patient_vals
    st.write("")
    
    # Input Form
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            st.markdown("""
            <div class="section-header">
                <span>👤 Demographics & Pressure</span>
            </div>
            """, unsafe_allow_html=True)
            in_age = st.slider("Patient Age (Years)", 25, 80, vals["age"])
            in_sex = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", index=0 if vals["sex"] == 1 else 1)
            in_trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, vals["trestbps"])
            in_chol = st.slider("Serum Cholesterol (mg/dl)", 120, 550, vals["chol"])
            
        with col2:
            st.markdown("""
            <div class="section-header">
                <span>💓 Symptoms & Resting ECG</span>
            </div>
            """, unsafe_allow_html=True)
            in_cp = st.selectbox(
                "Chest Pain Classification",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "Typical Angina (Pressure/Tightness)",
                    1: "Atypical Angina (Mild Discomfort)",
                    2: "Non-Anginal Discomfort",
                    3: "Asymptomatic"
                }[x],
                index=vals["cp"]
            )
            in_restecg = st.selectbox(
                "Resting ECG Findings",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "Normal Sinus Rhythm",
                    1: "ST-T Wave Abnormality",
                    2: "Left Ventricular Hypertrophy"
                }[x],
                index=vals["restecg"]
            )
            in_thalach = st.slider("Max Achieved Heart Rate (bpm)", 70, 210, vals["thalach"])
            in_fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Normal (<= 120)" if x == 0 else "Elevated (> 120)", index=vals["fbs"])

        with col3:
            st.markdown("""
            <div class="section-header">
                <span>⚡ Cardiac Stress Parameters</span>
            </div>
            """, unsafe_allow_html=True)
            in_exang = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=vals["exang"])
            in_oldpeak = st.slider("ST Depression (Exercise vs Rest)", 0.0, 6.0, float(vals["oldpeak"]), step=0.1)
            in_slope = st.selectbox("Peak Exercise ST Segment Slope", options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x], index=vals["slope"])
            in_ca = st.slider("Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, vals["ca"])
            in_thal = st.selectbox("Thallium Stress Test Result", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x], index=vals["thal"] - 1)

        st.write("")
        algo_c1, algo_c2 = st.columns([2, 1])
        with algo_c1:
            algo_choice = st.radio(
                "Choose Machine Learning Algorithm:",
                ["$k$-Nearest Neighbours ($k$-NN, $k=5$)", "Gaussian Naïve Bayes"],
                horizontal=True
            )
        with algo_c2:
            st.write("")
            btn_predict = st.form_submit_button("🔍 Run Diagnostic Assessment", type="primary", use_container_width=True)

    if btn_predict:
        model_key = 'knn' if 'k-NN' in algo_choice else 'nb'
        patient_record = {
            'age': in_age, 'sex': in_sex, 'cp': in_cp, 'trestbps': in_trestbps,
            'chol': in_chol, 'fbs': in_fbs, 'restecg': in_restecg, 'thalach': in_thalach,
            'exang': in_exang, 'oldpeak': in_oldpeak, 'slope': in_slope, 'ca': in_ca, 'thal': in_thal
        }
        
        result = predict_patient_risk(patient_record, model_choice=model_key)
        
        if result['prediction'] == 1:
            st.markdown(f"""
            <div class="result-card-high">
                <span class="res-badge badge-red">⚠️ Risk Alert</span>
                <div class="big-score">Elevated Cardiovascular Risk</div>
                <p style="font-size: 1.05rem; font-weight: 600; margin-bottom: 8px;">
                    Model Confidence: {result['confidence']*100:.1f}% &nbsp;|&nbsp; High Risk Probability: {result['probability_high_risk']*100:.1f}%
                </p>
                <p style="margin-bottom: 12px;">The patient vitals display significant clinical hallmarks consistent with coronary heart disease.</p>
                <div style="background: white; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #ef4444;">
                    <strong>Recommended Next Steps:</strong> Immediate cardiology consultation, stress echocardiogram, lipid lowering regimen, and coronary angiography evaluation.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-low">
                <span class="res-badge badge-green">✅ Low Risk</span>
                <div class="big-score">Healthy Physiological Profile</div>
                <p style="font-size: 1.05rem; font-weight: 600; margin-bottom: 8px;">
                    Model Confidence: {result['confidence']*100:.1f}% &nbsp;|&nbsp; Low Risk Probability: {result['probability_low_risk']*100:.1f}%
                </p>
                <p style="margin-bottom: 12px;">Clinical parameters are within favorable ranges with minimal likelihood of ischemic heart disease.</p>
                <div style="background: white; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #22c55e;">
                    <strong>Recommended Next Steps:</strong> Routine annual check-up, regular cardiovascular exercise, and maintenance of healthy diet.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.markdown("##### **Probability Breakdown**")
        pb_c1, pb_c2 = st.columns(2)
        with pb_c1:
            st.write(f"**Low Risk Probability:** `{result['probability_low_risk']*100:.1f}%`")
            st.progress(result['probability_low_risk'])
        with pb_c2:
            st.write(f"**High Risk Probability:** `{result['probability_high_risk']*100:.1f}%`")
            st.progress(result['probability_high_risk'])


# =====================================================================
# MODULE 2: CHEST X-RAY DIAGNOSIS (KERAS CNN)
# =====================================================================
with tab_image:
    st.markdown("#### **Automated Radiograph Image Analysis**")
    st.caption("Deep Convolutional Neural Network (CNN) for rapid detection of pulmonary opacity and pneumonia.")
    
    # Input selection
    input_type = st.radio(
        "Select Radiograph Source:",
        ["📁 Use Clinical Demonstration Scans", "📤 Upload Custom Radiograph File"],
        horizontal=True
    )
    
    active_image = None
    sample_desc = ""
    
    if input_type == "📁 Use Clinical Demonstration Scans":
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            if st.button("🟢 Load Normal Chest Radiograph", use_container_width=True):
                st.session_state.active_xray = "normal"
        with b_c2:
            if st.button("🔴 Load Pneumonia / Pathological Radiograph", use_container_width=True):
                st.session_state.active_xray = "pneumonia"
                
        if "active_xray" not in st.session_state:
            st.session_state.active_xray = "normal"
            
        if st.session_state.active_xray == "normal":
            img_path = os.path.join(BASE_DIR, "data", "samples", "sample_normal_chest_xray.png")
            sample_desc = "Clinical Reference: Clear bilateral lung fields with sharp diaphragmatic contours."
        else:
            img_path = os.path.join(BASE_DIR, "data", "samples", "sample_pneumonia_chest_xray.png")
            sample_desc = "Pathological Reference: Confluent alveolar consolidation and radiopaque pulmonary opacities."
            
        if os.path.exists(img_path):
            active_image = Image.open(img_path)
            
    else:
        uploaded_file = st.file_uploader("Upload Chest Radiograph (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file)
            sample_desc = f"Uploaded Scan: {uploaded_file.name}"

    st.write("")
    
    if active_image is not None:
        view_c1, view_c2 = st.columns([1, 1.25], gap="large")
        
        with view_c1:
            st.markdown("""
            <div class="section-header">
                <span>🩻 Radiograph View</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(active_image, caption=sample_desc, width=330)
            
        with view_c2:
            st.markdown("""
            <div class="section-header">
                <span>🔬 CNN Diagnostic Inference</span>
            </div>
            """, unsafe_allow_html=True)
            
            run_cnn = st.button("⚡ Run Keras CNN Inference", type="primary", use_container_width=True)
            
            if run_cnn:
                with st.spinner("Analyzing radiograph through convolutional feature maps..."):
                    pred = predict_single_image(active_image)
                    
                if pred["is_pathological"]:
                    st.markdown(f"""
                    <div class="result-card-high">
                        <span class="res-badge badge-red">🚨 Pathological Finding</span>
                        <div class="big-score">Pneumonia Detected</div>
                        <p style="font-size: 1.05rem; font-weight: 600; margin-bottom: 8px;">
                            Diagnostic Confidence: {pred['confidence']*100:.1f}%
                        </p>
                        <p style="margin-bottom: 12px;">Confluent radiopaque opacities and alveolar infiltrates identified in lung fields.</p>
                        <div style="background: white; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #ef4444;">
                            <strong>Recommended Protocol:</strong> Immediate clinical auscultation, sputum culture, pulse oximetry, and empirical antibiotic therapy.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-low">
                        <span class="res-badge badge-green">✅ Clear Radiograph</span>
                        <div class="big-score">Normal Pulmonary Field</div>
                        <p style="font-size: 1.05rem; font-weight: 600; margin-bottom: 8px;">
                            Diagnostic Confidence: {pred['confidence']*100:.1f}%
                        </p>
                        <p style="margin-bottom: 12px;">Clear radiolucent lung parenchyma; no acute consolidation or pleural effusion observed.</p>
                        <div style="background: white; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #22c55e;">
                            <strong>Recommended Protocol:</strong> No radiologically evident pulmonary infection. Correlate with upper-respiratory symptoms if persistent.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.write("")
                st.markdown("##### **Model Probability Distribution**")
                prob_c1, prob_c2 = st.columns(2)
                with prob_c1:
                    st.write(f"**Normal:** `{pred['probabilities']['Normal']*100:.1f}%`")
                    st.progress(pred['probabilities']['Normal'])
                with prob_c2:
                    st.write(f"**Pneumonia:** `{pred['probabilities']['Pneumonia']*100:.1f}%`")
                    st.progress(pred['probabilities']['Pneumonia'])
    else:
        st.info("Please select a sample scan or upload an image to begin diagnostic analysis.")
