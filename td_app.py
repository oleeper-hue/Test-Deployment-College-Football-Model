
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Test Deployment Model", page_icon="🔬")

HERE = Path(__file__).parent
MODEL_PATH = HERE / "td_model.pkl"
DATA_INFO_PATH = HERE / "td_data_info.pkl"

# ----- Helpers to load artifacts -----
@st.cache_resource
def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

# ----- Load model & data_info -----
try:
    model = load_pickle(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}.\n{e}")
    st.stop()

try:
    data_info = load_pickle(DATA_INFO_PATH)
except Exception as e:
    st.error(f"Could not load data_info at {DATA_INFO_PATH}.\n{e}")
    st.stop()

# ----- Read expected metadata -----
expected_columns = data_info.get("expected_columns", [])
feature_order = data_info.get("feature_order", [])
categorical_unique_values = data_info.get("categorical_unique_values", {})
label_encode_mapping = data_info.get("label_encode_mapping", {})  # e.g. {"Freshman":0,...} or code->label depending on how you saved it
label_encode_levels = data_info.get("label_encode_levels", [])
ohe_categorical_columns = data_info.get("ohe_categorical_columns", [])
numeric_ranges = data_info.get("numeric_ranges", {})

# If label_encode_mapping is code->label, invert it to label->code for UI convenience
# We expect label_encode_mapping to be label->code (e.g. {"Freshman":0,...})
# If it's reversed, try to infer and invert.
if label_encode_mapping and not isinstance(list(label_encode_mapping.values())[0], int):
    # strange mapping: assume code->label, invert it
    label_encode_mapping = {v: int(k) for k, v in label_encode_mapping.items()}

# ----- Page UI -----
st.title("Test Deployment Model: College Football Longest Throw Prediction")
st.caption("Enter player info below and click Predict")

# store UI selections here
ui_inputs = {}

# Build UI in the user-specified order
for feat in feature_order:
    # numeric
    if feat in numeric_ranges or feat not in categorical_unique_values:
        # treat as numeric if numeric_ranges defined OR not in categorical list
        rng = numeric_ranges.get(feat, {})
        lo = int(rng.get("min", 0))
        hi = int(rng.get("max", 1000))
        default = int(rng.get("default", (lo + hi) // 2))
        ui_inputs[feat] = st.slider(feat.replace("_", " ").title(), min_value=lo, max_value=hi, value=default)
    else:
        # categorical
        options = categorical_unique_values.get(feat, [])
        if len(options) == 0:
            # fallback: empty list -> text input
            ui_inputs[feat] = st.text_input(feat.replace("_", " ").title())
        else:
            ui_inputs[feat] = st.selectbox(feat.replace("_", " ").title(), options)

st.divider()

# ----- Convert UI inputs to a raw dataframe -----
raw_row = {}
for k, v in ui_inputs.items():
    raw_row[k] = v

raw_df = pd.DataFrame([raw_row])

# ----- Apply same encoding used in training -----
# 1) Label encoding for ordinal features (e.g., "class") if mapping present
# We assume label_encode_mapping maps label -> integer (e.g. "Freshman":0)
# If mapping is present and a column in raw_df matches keys, map it to "<col>_encoded"
for label_col, mapping in (("class", label_encode_mapping),):
    if mapping and label_col in raw_df.columns:
        # map label->code -> store as e.g. "class_encoded"
        raw_df[f"{label_col}_encoded"] = raw_df[label_col].map(mapping)
        # optionally keep original label column if you want; model expects encoded name
        # remove original label column if model did not use it (training used encoded col)
        if label_col in raw_df.columns:
            raw_df.drop(columns=[label_col], inplace=True)

# 2) One-hot encode specified categorical columns (drop_first=True to match training)
# Make sure to only apply OHE on columns that remain in raw_df
ohe_cols_to_apply = [c for c in ohe_categorical_columns if c in raw_df.columns]
if ohe_cols_to_apply:
    input_encoded = pd.get_dummies(raw_df, columns=ohe_cols_to_apply, drop_first=True, dtype=int)
else:
    input_encoded = raw_df.copy()

# If training used different column names (e.g., "class_encoded"), ensure they exist in input_encoded
# and align to expected_columns order.
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder to expected columns (this will also drop any unexpected columns)
input_encoded = input_encoded[expected_columns]

st.subheader("Prepared model input")
st.write(input_encoded)

# ----- Predict button -----
if st.button("Predict"):
    try:
        prediction = model.predict(input_encoded)
        # model.predict returns array-like; show first element
        if hasattr(prediction, "__len__"):
            pred_value = float(prediction[0])
        else:
            pred_value = float(prediction)

        st.subheader("Prediction Result")
        st.success(f"Predicted max throw: {round(pred_value, 2)}")
    except Exception as e:
        st.error(f"Inference failed: {e}")
