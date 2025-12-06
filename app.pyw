# -------------------------------------------------------
# FINAL WORKING app.py  (buttons fixed)
# -------------------------------------------------------

import streamlit as st
import numpy as np
import joblib
import random
from pathlib import Path

try:
    from tensorflow.keras.models import load_model
except:
    load_model = None

# -------------------------------------------
# CORRECT MODEL PATHS
# -------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
MODELS_DIR = (BASE_DIR / "models").resolve()

# -------------------------------------------
# LOAD MODELS
# -------------------------------------------
def safe_load_regression():
    possible = [
        "best_regression_model.pkl",
        "best_model.pkl",
        "best_model_ann.h5",
        "best_model.h5"
    ]
    for p in possible:
        path = MODELS_DIR / p
        if path.exists():
            if path.suffix == ".pkl":
                return joblib.load(path), "sklearn"
            if path.suffix == ".h5" and load_model is not None:
                return load_model(str(path)), "keras"
    return None, None

warnings_list = []
MODELS_LOADED = True

try: encoder = joblib.load(MODELS_DIR / "encoder.pkl")
except: encoder=None; warnings_list.append("encoder.pkl missing!")

try: scaler = joblib.load(MODELS_DIR / "scaler.pkl")
except: scaler=None; warnings_list.append("scaler.pkl missing!")

reg_model, reg_type = safe_load_regression()
if reg_model is None:
    warnings_list.append("Regression model missing!")

try: cluster_encoder = joblib.load(MODELS_DIR / "cluster_encoder.pkl")
except: cluster_encoder=None; warnings_list.append("cluster_encoder.pkl missing!")

try: cluster_scaler = joblib.load(MODELS_DIR / "cluster_scaler.pkl")
except: cluster_scaler=None; warnings_list.append("cluster_scaler.pkl missing!")

try: cluster_model = joblib.load(MODELS_DIR / "clustering_model.pkl")
except: cluster_model=None; warnings_list.append("clustering_model.pkl missing!")

try: cluster_names = joblib.load(MODELS_DIR / "cluster_names.pkl")
except: cluster_names=None; warnings_list.append("cluster_names.pkl missing!")

if warnings_list:
    MODELS_LOADED = False

# -------------------------------------------
# INPUT OPTIONS
# -------------------------------------------
cut_list = ["Fair","Good","Very Good","Premium","Ideal"]
color_list = ["D","E","F","G","H","I","J"]
clarity_list = ["IF","VVS1","VVS2","VS1","VS2","SI1","SI2","I1"]

def random_values():
    return {
        "carat": round(random.uniform(0.3, 1.5),2),
        "depth": round(random.uniform(60, 63.5),1),
        "table": round(random.uniform(55, 60),1),
        "x": round(random.uniform(4, 7.5),2),
        "y": round(random.uniform(4, 7.5),2),
        "z": round(random.uniform(2.5, 5),2),
        "cut": random.choice(cut_list),
        "color": random.choice(color_list),
        "clarity": random.choice(clarity_list)
    }

# -------------------------------------------
# STREAMLIT UI
# -------------------------------------------
st.title("💎 Diamond Price Predictor & Market Segmentation")

# ---- Auto Predict Button ----
if st.button("✨ Auto-Predict"):
    vals = random_values()
    for k, v in vals.items():
        st.session_state[k] = v

# Ensure defaults exist
for k, v in random_values().items():
    st.session_state.setdefault(k, v)

# ---- Inputs ----
col1, col2, col3 = st.columns(3)

with col1:
    carat = st.number_input("Carat", 0.1, 5.0, value=float(st.session_state["carat"]), step=0.01, key="carat")
    depth = st.number_input("Depth (%)", 45.0, 80.0, value=float(st.session_state["depth"]), step=0.1, key="depth")
    x = st.number_input("x (mm)", 0.1, 15.0, value=float(st.session_state["x"]), step=0.01, key="x")

with col2:
    cut = st.selectbox("Cut", cut_list, index=cut_list.index(st.session_state["cut"]), key="cut")
    table = st.number_input("Table (%)", 45.0, 80.0, value=float(st.session_state["table"]), step=0.1, key="table")
    y = st.number_input("y (mm)", 0.1, 15.0, value=float(st.session_state["y"]), step=0.01, key="y")

with col3:
    color = st.selectbox("Color", color_list, index=color_list.index(st.session_state["color"]), key="color")
    clarity = st.selectbox("Clarity", clarity_list, index=clarity_list.index(st.session_state["clarity"]), key="clarity")
    z = st.number_input("z (mm)", 0.1, 15.0, value=float(st.session_state["z"]), step=0.01, key="z")

# -------------------------------------------
# PREDICTION
# -------------------------------------------
if st.button("🔮 Predict Price & Cluster"):

    if not MODELS_LOADED:
        st.error("❌ Models not loaded!")
    else:
        try:
            volume = x * y * z

            raw = np.array([[carat, cut, color, clarity, depth, table, x, y, z, volume]], dtype=object)

            # ---------------- Regression ----------------
            reg_cat_idx = [1,2,3]
            reg_num_idx = [0,4,5,7,8,9]

            enc = encoder.transform(raw[:, reg_cat_idx])
            scl = scaler.transform(raw[:, reg_num_idx])
            reg_input = np.hstack([scl, enc]).astype(float)

            if reg_type == "sklearn":
                log_price = reg_model.predict(reg_input)[0]
            else:
                log_price = float(reg_model.predict(reg_input).flatten()[0])

            price_inr = np.expm1(log_price)
            price_usd = price_inr / 85

            st.success(f"💰 INR: ₹ {price_inr:,.2f}")
            st.success(f"💰 USD: $ {price_usd:,.2f}")

            # ---------------- Clustering ----------------
            cluster_num_idx = [0,4,5,7,8,9]
            cluster_cat_idx = [1,2,3]

            cc = cluster_encoder.transform(raw[:, cluster_cat_idx])
            cs = cluster_scaler.transform(raw[:, cluster_num_idx])
            cluster_input = np.hstack([cs, cc]).astype(float)

            cid = int(cluster_model.predict(cluster_input)[0])
            cname = cluster_names.get(cid, "Unknown")

            st.info(f"💎 Cluster {cid} → {cname}")

        except Exception as e:
            st.error("❌ Prediction Error")
            st.exception(e)

