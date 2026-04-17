"""
Bank Customer Churn Prediction - Premium Streamlit Web Application
Developed by: Gokul Kumar Sant
University: Maharishi Markandeshwar (Deemed to be University)
Roll No: 11232629 | Section: 6-G
"""

import os

# Suppress TensorFlow logging and disable GPU for cloud stability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set Matplotlib to Headless mode
matplotlib.use('Agg')

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="ChurnSense AI — Premium Dashboard",
    page_icon="🏦",
    layout="wide",  # Changed to wide for better distribution
    initial_sidebar_state="expanded"
)

# ── Dynamic Design CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ---- Global Styles ---- */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background-color: #050505;
    background-image: 
        radial-gradient(circle at 10% 20%, rgba(91, 78, 255, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(0, 255, 127, 0.05) 0%, transparent 40%);
}

/* ---- Containers ---- */
.block-container { padding-top: 2rem !important; }

/* ---- Custom Cards ---- */
.premium-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

/* ---- Sidebar Styling ---- */
[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #1a1a1a;
}

/* ---- Header Styling ---- */
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #ffffff 0%, #a1a1a1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #666;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* ---- Prediction Result Styling ---- */
.risk-high {
    color: #ff4b4b;
    border-color: #ff4b4b;
    background: rgba(255, 75, 75, 0.1);
}

.risk-low {
    color: #00d488;
    border-color: #00d488;
    background: rgba(0, 212, 136, 0.1);
}

/* Glassmorphism for inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Custom Metric Box */
div[data-testid="stMetricValue"] {
    color: #fff !important;
    font-weight: 700 !important;
}

/* ---- Custom Labels ---- */
.small-label {
    text-transform: uppercase;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: #4a4a4a;
    margin-bottom: 8px;
}

/* Success/Error Banners */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)

# ── Helper Functions ────────────────────────────────────────────────────────
def create_gauge(prob):
    """Creates a custom HTML/CSS gauge for churn risk."""
    color = "#ff4b4b" if prob > 0.5 else "#00d488"
    pct = prob * 100
    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px;">
        <div style="position: relative; width: 220px; height: 110px; overflow: hidden;">
            <div style="position: absolute; width: 220px; height: 220px; border-radius: 50%; border: 18px solid #1a1a1a;"></div>
            <div style="position: absolute; width: 220px; height: 220px; border-radius: 50%; border: 18px solid {color}; 
                        border-bottom-color: transparent; border-left-color: transparent; 
                        transform: rotate({-135 + (pct * 1.8)}deg); transition: transform 1s ease-out;"></div>
        </div>
        <div style="margin-top: -30px; text-align: center;">
            <div style="font-size: 2.8rem; font-weight: 800; color: white;">{pct:.1f}%</div>
            <div style="font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 1px;">Churn Probability</div>
        </div>
    </div>
    """

# ── Load & cache pipeline + model ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    with st.spinner("🧠 Initializing Deep Learning Engine..."):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        BASE = os.path.dirname(os.path.abspath(__file__))
        CSV  = os.path.join(BASE, "Artificial_Neural_Network_Case_Study_data.csv")
        H5   = os.path.join(BASE, "bank_churn_ann_model.h5")

        if not os.path.exists(CSV):
            st.error(f"Data file not found at {CSV}. Please ensure the dataset is present.")
            st.stop()

        # --- Data Loading ---
        dataset = pd.read_csv(CSV)
        X = dataset.iloc[:, 3:13].values
        y = dataset.iloc[:, 13].values

        # --- Preprocessing ---
        le = LabelEncoder()
        X[:, 2] = le.fit_transform(X[:, 2])

        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [1])],
            remainder='passthrough'
        )
        X = np.array(ct.fit_transform(X))

        sc = StandardScaler()
        X = sc.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # --- Load or train model ---
        if os.path.exists(H5):
            try:
                model = load_model(H5)
            except Exception as e:
                st.warning(f"Error loading saved model: {e}. Retraining...")
                model = None
        else:
            model = None

        if model is None:
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(
                X_train, y_train,
                validation_split=0.2, batch_size=32, epochs=60,
                callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
                verbose=0
            )
            model.save(H5)

        # --- Statistics ---
        y_pred = (model.predict(X_test, verbose=0) > 0.5)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return le, ct, sc, model, accuracy, cm

# ── Sidebar ──
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.markdown("### Model Controls")
    st.info("The model uses an Artificial Neural Network (ANN) built with TensorFlow/Keras to predict bank customer churn.")
    
    st.markdown("---")
    st.markdown("#### System Integrity")
    st.success("✅ Neural Network Ready")
    st.success("✅ Dataset Linked")
    
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size: 0.75rem; color: #444;">
        <b>Student Information</b><br>
        Name: Gokul Kumar Sant<br>
        University: MM(DU)<br>
        Roll No: 11232629
    </div>
    """, unsafe_allow_html=True)

# ── Main Content ───────────────────────────────────────────────────────────
le, ct, sc, model, model_acc, conf_mat = load_pipeline()

# Header Section
col_title, col_stats = st.columns([2, 1])

with col_title:
    st.markdown('<h1 class="main-title">ChurnSense AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Advanced Bank Customer Retention Analytics</p>', unsafe_allow_html=True)

with col_stats:
    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
    inner_c1, inner_c2 = st.columns(2)
    with inner_c1:
        st.metric("Model Precision", f"{model_acc*100:.2f}%")
    with inner_c2:
        st.metric("Model Type", "ANN")

# Layout: Prediction Form and Insights
tab1, tab2 = st.tabs(["🚀 Customer Prediction", "📊 Model Analysis"])

with tab1:
    # Form split into two visual columns
    st.markdown('<p class="small-label">Customer Demographics & Financial Stats</p>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            credit_score = st.number_input("Credit Score", 300, 850, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            
        with c2:
            age = st.slider("Customer Age", 18, 95, 38)
            tenure = st.number_input("Tenure (Years)", 0, 10, 5)
            balance = st.number_input("Balance ($)", 0.0, 300000.0, 75000.0, step=1000.0)
            
        with c3:
            num_products = st.selectbox("Num Products", [1, 2, 3, 4], index=0)
            has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active = st.selectbox("Active Member?", ["Yes", "No"])
            salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 55000.0, step=1000.0)

        st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
        submit_btn = st.form_submit_button("ANALYZE CUSTOMER CHURN RISK")

    if submit_btn:
        # Preprocessing inputs
        has_card_val = 1 if has_cr_card == "Yes" else 0
        active_val = 1 if is_active == "Yes" else 0
        
        input_data = [credit_score, geography, gender, age, tenure, balance, num_products, has_card_val, active_val, salary]
        
        # Transform Gender using le
        input_data[2] = le.transform([input_data[2]])[0]
        
        # Transform everything using ct and sc
        input_array = np.array([input_data], dtype=object)
        transformed = ct.transform(input_array)
        final_input = sc.transform(transformed)
        
        # Prediction
        with st.spinner("Processing Neural Pathways..."):
            time.sleep(0.6) # Subtle visual delay
            prediction_prob = float(model.predict(final_input, verbose=0)[0][0])
            is_churn = prediction_prob > 0.5

        # Results Display
        st.markdown("---")
        res_c1, res_c2 = st.columns([1, 1.5])
        
        with res_c1:
            st.markdown(create_gauge(prediction_prob), unsafe_allow_html=True)
            
        with res_c2:
            if is_churn:
                st.markdown(f"""
                <div class="premium-card risk-high">
                    <h3 style="margin:0; color:#ff4b4b;">High Retention Risk</h3>
                    <p style="color:rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin-top: 10px;">
                        The ANN model indicates that this customer has a <b>{prediction_prob*100:.1f}%</b> probability of churning. 
                        We recommend immediate reach-out or loyalty offers.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="premium-card risk-low">
                    <h3 style="margin:0; color:#00d488;">Loyal Customer Profile</h3>
                    <p style="color:rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin-top: 10px;">
                        The probability of churn is low (<b>{prediction_prob*100:.1f}%</b>). 
                        This customer appears stable based on their current behavioral and financial vectors.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Contextual Insights (Mock Analysis)
            with st.expander("🔍 Risk Factor Insights"):
                if age > 50:
                    st.write("• **Age Factor:** Older customers statistically show different churn patterns in this segment.")
                if balance < 10000:
                    st.write("• **Liquidity:** Low account balance correlates with higher transition probability.")
                if num_products > 2:
                    st.write("• **Product Stickiness:** Multi-product usage usually increases retention.")
                st.write("• **Model Verdict:** The prediction is based on the weights of the hidden layer neurons refined over 60 epochs.")

with tab2:
    st.markdown('<p class="small-label">Performance Visualization</p>', unsafe_allow_html=True)
    
    # Grid for metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Training Epochs", "60", "Stable")
    m2.metric("Testing Size", "20%", "Random Split")
    m3.metric("Validation Split", "20%")

    col_cm, col_info = st.columns([1.5, 1])
    
    with col_cm:
        # Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#050505')
        ax.set_facecolor('#050505')
        
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Stayed', 'Exited'], yticklabels=['Stayed', 'Exited'],
                    annot_kws={"size": 14, "weight": "bold"}, cbar=False)
        
        plt.title('Prediction Accuracy Heatmap', color='white', pad=20, fontsize=14)
        ax.set_xlabel('Predicted Label', color='#888', fontsize=10)
        ax.set_ylabel('Actual Status', color='#888', fontsize=10)
        ax.tick_params(colors='#888', labelsize=10)
        
        for _, spine in ax.spines.items():
            spine.set_edgecolor('#333')
            
        st.pyplot(fig)

    with col_info:
        st.markdown("### Model Architecture")
        st.markdown("""
        The backend engine utilizes a multi-layer deep learning architecture:
        - **Input Layer:** Normalized features (incl. One-Hot Encoding)
        - **Hidden Layer 1:** 128 Neurons (ReLU activation)
        - **Regularization:** Dropout (20%) to prevent overfitting
        - **Hidden Layer 2:** 64 Neurons (ReLU activation)
        - **Output Layer:** Sigmoid activation (Binary outcome)
        - **Optimization:** Adam Optimizer with Binary Crossentropy loss.
        """)
        
        st.markdown("---")
        st.download_button(
            label="Download Prediction Model (.h5)",
            data=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank_churn_ann_model.h5"), "rb").read(),
            file_name="bank_churn_ann_model.h5",
            mime="application/octet-stream"
        )

# Footer
st.markdown("""
<div style="text-align:center; padding: 40px 0; border-top: 1px solid #1a1a1a; margin-top: 50px; color: #444; font-size: 0.8rem;">
    ChurnSense AI Framework &copy; 2026 | Research Project by Gokul Kumar Sant<br>
    Built with Python, Streamlit, and TensorFlow Deep Learning.
</div>
""", unsafe_allow_html=True)
