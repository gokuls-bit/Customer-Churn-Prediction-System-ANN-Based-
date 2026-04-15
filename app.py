"""
Bank Customer Churn Prediction - Flask Web Application
Developed by: Gokul Kumar Sant
University: Maharishi Markandeshwar (Deemed to be University)
Roll No: 11232629 | Section: 6-G
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# FLASK APP INITIALIZATION
# ============================================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bank_churn_ann_model.h5")

# ============================================================
# LOAD DATA & FIT PREPROCESSORS
# ============================================================
print("[INFO] Loading dataset...")
dataset = pd.read_csv(os.path.join(BASE_DIR, "Artificial_Neural_Network_Case_Study_data.csv"))
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# --- Step 1: Label Encode Gender (column index 2) ---
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# --- Step 2: One Hot Encode Geography (column index 1) ---
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))

# --- Step 3: Standard Scale all features ---
sc = StandardScaler()
X = sc.fit_transform(X)

# --- Step 4: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# LOAD OR TRAIN THE MODEL
# ============================================================
# Import TensorFlow here (after data prep) for cleaner output
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

if os.path.exists(MODEL_PATH):
    print(f"[OK] Loading existing model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
else:
    print("[!] Model file not found. Training a new model...")
    print("[*] This will take 1-2 minutes...")

    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=100,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"[OK] Model trained and saved to {MODEL_PATH}")

# ============================================================
# COMPUTE EVALUATION METRICS
# ============================================================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5)
model_accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# --- Generate Confusion Matrix Image ---
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
cm_path = os.path.join(BASE_DIR, "static", "confusion_matrix.png")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Stay', 'Leave'], yticklabels=['Stay', 'Leave'])
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(cm_path, dpi=150)
plt.close(fig)

print(f"[OK] Model ready. Test Accuracy: {model_accuracy*100:.2f}%")


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    """Render the home page with the input form."""
    return render_template(
        "index.html",
        accuracy=f"{model_accuracy * 100:.2f}"
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction request from the form."""
    try:
        # --- Collect form data ---
        credit_score = int(request.form["credit_score"])
        geography = request.form["geography"]
        gender = request.form["gender"]
        age = int(request.form["age"])
        tenure = int(request.form["tenure"])
        balance = float(request.form["balance"])
        num_products = int(request.form["num_products"])
        has_cr_card = 1 if request.form["has_cr_card"] == "Yes" else 0
        is_active = 1 if request.form["is_active"] == "Yes" else 0
        estimated_salary = float(request.form["estimated_salary"])

        # --- Build feature list (same order as training) ---
        new_data = [
            credit_score, geography, gender, age, tenure,
            balance, num_products, has_cr_card, is_active, estimated_salary
        ]

        # --- Apply LabelEncoder to Gender (index 2) ---
        new_data[2] = le.transform([new_data[2]])[0]

        # --- Apply OneHotEncoder for Geography via ColumnTransformer ---
        new_data_np = np.array([new_data], dtype=object)
        new_data_encoded = ct.transform(new_data_np)

        # --- Apply StandardScaler ---
        new_data_scaled = sc.transform(new_data_encoded)

        # --- Predict ---
        prediction_prob = float(model.predict(new_data_scaled)[0][0])
        churn = prediction_prob > 0.5

        return render_template(
            "result.html",
            probability=f"{prediction_prob * 100:.1f}",
            churn=churn,
            accuracy=f"{model_accuracy * 100:.2f}",
            credit_score=credit_score,
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            balance=balance,
            num_products=num_products,
            has_cr_card="Yes" if has_cr_card else "No",
            is_active="Yes" if is_active else "No",
            estimated_salary=estimated_salary
        )
    except Exception as e:
        return render_template(
            "index.html",
            accuracy=f"{model_accuracy * 100:.2f}",
            error=f"Prediction failed: {str(e)}"
        )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
