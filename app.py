"""
Bank Customer Churn Prediction - Streamlit Web Application
Developed by: Gokul Kumar Sant
University: Maharishi Markandeshwar (Deemed to be University)
Roll No: 11232629 | Section: 6-G
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Metallic Black CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background-color: #0a0a0a;
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(60,60,60,0.10) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 100%, rgba(40,40,40,0.08) 0%, transparent 60%);
}

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem; max-width: 740px; }

/* ---- Inputs & Selects ---- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child {
    background: #181818 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    color: #e8e8e8 !important;
    transition: border-color .2s, box-shadow .2s;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:first-child:focus-within {
    border-color: #3d3d3d !important;
    box-shadow: 0 0 0 3px rgba(120,120,120,.10) !important;
}
input, .stSelectbox * { color: #e8e8e8 !important; }

/* ---- Labels ---- */
label, .stSlider label {
    color: #4a4a4a !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: .8px !important;
    text-transform: uppercase !important;
}

/* ---- Slider ---- */
.stSlider > div > div > div > div {
    background: #3d3d3d !important;
}

/* ── Predict Button ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 100%) !important;
    border: 1px solid #3a3a3a !important;
    border-top-color: #4a4a4a !important;
    border-radius: 12px !important;
    color: #d4d4d4 !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 14px !important;
    margin-top: 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,.6), 0 1px 0 rgba(255,255,255,.04) inset !important;
    transition: all .2s ease !important;
}
div.stButton > button:hover {
    background: linear-gradient(180deg, #333 0%, #222 100%) !important;
    border-color: #555 !important;
    color: #fff !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 28px rgba(0,0,0,.7) !important;
}

/* ── Divider ── */
hr { border-color: #1e1e1e !important; margin: 1.6rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #8a8a8a !important; }
</style>
""", unsafe_allow_html=True)


# ── Load & cache pipeline + model ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    BASE = os.path.dirname(os.path.abspath(__file__))
    CSV  = os.path.join(BASE, "Artificial_Neural_Network_Case_Study_data.csv")
    H5   = os.path.join(BASE, "bank_churn_ann_model.h5")

    # --- Data ---
    dataset = pd.read_csv(CSV)
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    # --- Preprocessing (must match training exactly) ---
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
        model = load_model(H5)
    else:
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(
            X_train, y_train,
            validation_split=0.2, batch_size=32, epochs=100,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )
        model.save(H5)

    # --- Metrics ---
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = (y_pred_prob > 0.5)
    accuracy    = accuracy_score(y_test, y_pred)
    cm          = confusion_matrix(y_test, y_pred)

    return le, ct, sc, model, accuracy, cm


# ── Boot ───────────────────────────────────────────────────────────────────
with st.spinner("Initializing ANN engine..."):
    le, ct, sc, model, model_acc, conf_mat = load_pipeline()

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding: 10px 0 28px;">
    <div style="display:inline-flex; align-items:center; gap:8px;
                background: linear-gradient(135deg,#1c1c1c,#252525);
                border:1px solid #3d3d3d; border-radius:50px;
                padding:6px 18px; font-size:.7rem; font-weight:600;
                letter-spacing:1.5px; text-transform:uppercase; color:#8a8a8a;
                margin-bottom:16px;">
        <span style="width:6px;height:6px;background:#2ecc71;border-radius:50%;
                     box-shadow:0 0 6px #2ecc71; display:inline-block;"></span>
        ANN Model Active
    </div>
    <h1 style="font-size:2.1rem; font-weight:800; letter-spacing:-.5px; margin:0;
               background: linear-gradient(180deg,#d4d4d4 0%,#8a8a8a 100%);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               background-clip:text;">
        ChurnSense AI
    </h1>
    <p style="color:#4a4a4a; font-size:.9rem; margin:6px 0 12px;">
        Bank Customer Churn Prediction System
    </p>
    <div style="color:#4a4a4a; font-size:.78rem; letter-spacing:.4px;">
        Model Accuracy &nbsp;·&nbsp;
        <span style="color:#d4d4d4; font-weight:700;">{model_acc*100:.2f}%</span>
        &nbsp;·&nbsp; ANN · TensorFlow
    </div>
</div>
""", unsafe_allow_html=True)

# ── Separator ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="height:1px; background:linear-gradient(90deg,transparent,#2a2a2a,transparent);
            margin-bottom:24px;"></div>
""", unsafe_allow_html=True)

# ── Section label ──────────────────────────────────────────────────────────
st.markdown("""
<p style="font-size:.68rem; font-weight:700; letter-spacing:2px; text-transform:uppercase;
          color:#333; border-bottom:1px solid #1e1e1e; padding-bottom:10px; margin-bottom:20px;">
    Customer Profile Input
</p>
""", unsafe_allow_html=True)

# ── Input Form ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    credit_score   = st.number_input("Credit Score",    min_value=300, max_value=850, value=600)
    geography      = st.selectbox("Geography",          ["France", "Germany", "Spain"])
    gender         = st.selectbox("Gender",             ["Male", "Female"])
    age            = st.slider("Age",                   min_value=18, max_value=100, value=40)
    tenure         = st.number_input("Tenure (Years)",  min_value=0,  max_value=10,  value=3)

with col2:
    balance        = st.number_input("Account Balance ($)",  min_value=0.0,  value=60000.0, step=1000.0)
    num_products   = st.selectbox("Number of Products",      [1, 2, 3, 4], index=1)
    has_cr_card    = st.selectbox("Has Credit Card?",         ["Yes", "No"])
    is_active      = st.selectbox("Active Member?",           ["Yes", "No"])
    salary         = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

# ── Predict Button ─────────────────────────────────────────────────────────
if st.button("▶   Run Churn Prediction"):
    with st.spinner("Analyzing through neural network..."):
        # Map binary
        has_card_val  = 1 if has_cr_card == "Yes" else 0
        is_active_val = 1 if is_active   == "Yes" else 0

        # Build raw feature vector
        raw = [credit_score, geography, gender, age, tenure,
               balance, num_products, has_card_val, is_active_val, salary]

        # Apply transformations
        raw[2]   = le.transform([raw[2]])[0]
        raw_np   = np.array([raw], dtype=object)
        encoded  = ct.transform(raw_np)
        scaled   = sc.transform(encoded)

        prob = float(model.predict(scaled, verbose=0)[0][0])
        churn = prob > 0.5

    # ── Result Banner ──────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)

    if churn:
        st.markdown(f"""
        <div style="background:#1a0808; border:1px solid #5a1515; border-radius:14px;
                    padding:28px 24px; text-align:center; margin-bottom:24px;">
            <div style="font-size:2.6rem; margin-bottom:8px;">⚠</div>
            <div style="font-size:.68rem; font-weight:700; letter-spacing:2.5px;
                        text-transform:uppercase; color:#ff4545; margin-bottom:6px;">
                High Risk
            </div>
            <div style="font-size:1.5rem; font-weight:800; color:#ff6b6b; margin-bottom:16px;">
                Customer Likely to Leave
            </div>
            <div style="display:inline-flex; align-items:baseline; gap:4px;
                        background:rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.07);
                        border-radius:50px; padding:10px 28px;">
                <span style="font-size:2rem; font-weight:800; color:#ff4545; line-height:1;">
                    {prob*100:.1f}%
                </span>
                <span style="font-size:.75rem; color:#7a7a7a; font-weight:500;">
                    Churn Probability
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#081a0e; border:1px solid #155a2a; border-radius:14px;
                    padding:28px 24px; text-align:center; margin-bottom:24px;">
            <div style="font-size:2.6rem; margin-bottom:8px;">✓</div>
            <div style="font-size:.68rem; font-weight:700; letter-spacing:2.5px;
                        text-transform:uppercase; color:#2ecc71; margin-bottom:6px;">
                Low Risk
            </div>
            <div style="font-size:1.5rem; font-weight:800; color:#4ade80; margin-bottom:16px;">
                Customer Likely to Stay
            </div>
            <div style="display:inline-flex; align-items:baseline; gap:4px;
                        background:rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.07);
                        border-radius:50px; padding:10px 28px;">
                <span style="font-size:2rem; font-weight:800; color:#2ecc71; line-height:1;">
                    {prob*100:.1f}%
                </span>
                <span style="font-size:.75rem; color:#7a7a7a; font-weight:500;">
                    Churn Probability
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Customer Profile Table ─────────────────────────────────────────────
    st.markdown("""
    <p style="font-size:.68rem; font-weight:700; letter-spacing:2px; text-transform:uppercase;
              color:#333; border-bottom:1px solid #1e1e1e; padding-bottom:10px; margin-bottom:16px;">
        Customer Profile
    </p>""", unsafe_allow_html=True)

    pcol1, pcol2 = st.columns(2)
    def profile_row(label, value):
        return f"""
        <div style="background:#181818; border:1px solid #222; border-radius:8px;
                    padding:10px 14px; display:flex; justify-content:space-between;
                    align-items:center; margin-bottom:6px;">
            <span style="font-size:.7rem; font-weight:600; text-transform:uppercase;
                         letter-spacing:.5px; color:#333;">{label}</span>
            <span style="font-size:.88rem; font-weight:600; color:#d4d4d4;">{value}</span>
        </div>"""

    with pcol1:
        st.markdown(profile_row("Credit Score",  credit_score),         unsafe_allow_html=True)
        st.markdown(profile_row("Geography",     geography),             unsafe_allow_html=True)
        st.markdown(profile_row("Gender",        gender),                unsafe_allow_html=True)
        st.markdown(profile_row("Age",           f"{age} yrs"),          unsafe_allow_html=True)
        st.markdown(profile_row("Tenure",        f"{tenure} yrs"),       unsafe_allow_html=True)

    with pcol2:
        st.markdown(profile_row("Balance",       f"${balance:,.0f}"),    unsafe_allow_html=True)
        st.markdown(profile_row("Products",      num_products),          unsafe_allow_html=True)
        st.markdown(profile_row("Credit Card",   has_cr_card),           unsafe_allow_html=True)
        st.markdown(profile_row("Active Member", is_active),             unsafe_allow_html=True)
        st.markdown(profile_row("Est. Salary",   f"${salary:,.0f}"),     unsafe_allow_html=True)

# ── Model Analytics (expandable) ───────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("📊 Model Performance — Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greys', ax=ax,
                xticklabels=['Stay', 'Leave'], yticklabels=['Stay', 'Leave'],
                linewidths=0.5, linecolor='#222')
    ax.set_xlabel("Predicted", color='#8a8a8a', fontsize=10)
    ax.set_ylabel("Actual",    color='#8a8a8a', fontsize=10)
    ax.tick_params(colors='#8a8a8a')
    for _, spine in ax.spines.items():
        spine.set_edgecolor('#2a2a2a')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(f"<p style='text-align:center; color:#4a4a4a; font-size:.8rem;'>Test Accuracy: <span style='color:#d4d4d4; font-weight:700;'>{model_acc*100:.2f}%</span></p>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 16px;
            border-top:1px solid #1a1a1a; margin-top:32px;
            font-size:.75rem; color:#333;">
    Developed by <span style="color:#4a4a4a; font-weight:600;">Gokul Kumar Sant</span>
    &nbsp;·&nbsp; MM(DU) &nbsp;·&nbsp; Roll: 11232629 &nbsp;·&nbsp; 6-G
</div>
""", unsafe_allow_html=True)
