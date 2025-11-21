from pathlib import Path
import streamlit as st
import pickle
import pandas as pd

# (Optional) helps unpickling but not strictly required if sklearn is installed
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Test Deployment Model", page_icon="🔬")

# ---------- Paths ----------
HERE = Path(__file__).parent
MODEL_PATH = HERE / "td_model.pkl"
DATA_INFO_PATH = HERE / "td_data_info.pkl"

# ---------- Load artifacts ----------
@st.cache_resource
def load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)

try:
    model = load_pickle(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}.\n{e}")
    st.stop()

try:
    data_info = load_pickle(DATA_INFO_PATH)
except Exception as e:
    st.error(
        f"Could not load data_info at {DATA_INFO_PATH}.\n"
        f"Ensure data_info.pkl exists and includes expected_columns.\n{e}"
    )
    st.stop()

expected_columns = data_info["expected_columns"]
numeric_ranges = data_info.get("numeric_ranges", {})

# ---------- Code↔Label maps ----------
class_levels = ["Freshman", "Sophomore", "Junior", "Senior"]
class_ord = {
    "Freshman": 0,
    "Sophomore": 1,
    "Junior": 2,
    "Senior": 3,
}

throwing_arm_map = {
    "L": "Left",
    "R": "Right",
}

def label_to_code(selection_label: str, mapping: dict) -> str:
    inv = {v: k for k, v in mapping.items()}
    return inv[selection_label]

# ---------- UI ----------
st.title("Test Deployment Model: College Football Longest Throw Prediction")
st.caption("This is the caption under the title")

st.header("Enter College Football Player's Information")

def num_slider(name, default, lo, hi, step=1):
    r = numeric_ranges.get(name, {})
    lo = int(r.get("min", lo))
    hi = int(r.get("max", hi))
    val = int(r.get("default", default))
    return st.slider(name.replace("_", " ").title(), min_value=lo, max_value=hi, value=val, step=step)

# Numeric features
max_bench = num_slider("max_bench", 225, 100, 405)

st.subheader("Beyond this label: categorical feature dropdown selections")

throwing_arm_label = st.selectbox("Throwing Arm", list(throwing_arm_map.values()))
throwing_arm = label_to_code(throwing_arm_label, throwing_arm_map)

class_label = st.selectbox("Class", class_levels)
Class = class_ord[class_label]

# ---------- Build raw row ----------
raw_row = {
    "max_bench": max_bench,
    "class": Class,
    "throwing_arm": throwing_arm
}

raw_df = pd.DataFrame([raw_row])

# ---------- Encode EXACTLY like training ----------
ohe_cols = ["throwing_arm"]

input_encoded = pd.get_dummies(raw_df, columns=ohe_cols, drop_first=True, dtype=int)

# 🔧 FIX 1 — Ensure original categorical column is removed
# (because the trained model never saw it)
if "throwing_arm" in input_encoded.columns:
    input_encoded = input_encoded.drop(columns=["throwing_arm"])

# 🔧 FIX 2 — Add any missing training-time columns (e.g., throwing_arm_R)
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Preserve correct column order
input_encoded = input_encoded[expected_columns]

print(input_encoded)

st.divider()
if st.button("Predict"):
    try:
        pred = model.predict(input_encoded)[0]

        st.subheader("Prediction Result")
        st.success(f"Prediction: {round(pred, 2)}")

    except Exception as e:
        st.error(f"Inference failed: {e}")
