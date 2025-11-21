from pathlib import Path
import streamlit as st
import pickle
import pandas as pd

# (Optional) helps unpickling but not strictly required if sklearn is installed
from sklearn.linear_model import LinearRegression #🚩 change model import from sklearn to intended model

st.set_page_config(page_title="Test Deployment Model", page_icon="🔬") #🚩change title on tab

# ---------- Paths ----------
HERE = Path(__file__).parent
MODEL_PATH = HERE / "td_model.pkl"   # your pruned tree
DATA_INFO_PATH = HERE / "td_data_info.pkl"         # must contain expected_columns, etc.

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

# These lists are only used to make nicer sliders; they won't change encoding
numeric_ranges = data_info.get("numeric_ranges", {})

# ---------- Code↔Label maps (UI shows labels; encoding uses codes) ----------

# Ordinal mapping for class (training used ordinal, not OHE) #🚩if no label encoding you can delete these 8 lines
class_levels = ["Freshman", "Sophomore", "Junior", "Senior"] #🚩list of desired dropdown values, change variable name for meaning
class_ord = { #🚩variable name for meaning
    "Freshman": 0, #🚩change to format {value in post-label encoding column: desired appearance on app}
    "Sophomore": 1, #🚩change to format {value in post-label encoding column: desired appearance on app}
    "Junior": 2, #🚩change to format {value in post-label encoding column: desired appearance on app}
    "Senior": 3, #🚩change to format {value in post-label encoding column: desired appearance on app}
}

# Dummy encoded categorical variables
throwing_arm_map = { #🚩change variable name for meaning
    "L": "Left", #🚩change to format {value in post-label encoding column: desired appearance on app}
    "R": "Right", #🚩change to format {value in post-label encoding column: desired appearance on app}
}

# Helper: label->code for UI selections
def label_to_code(selection_label: str, mapping: dict) -> str:
    # mapping is code->label; invert to label->code
    inv = {v: k for k, v in mapping.items()}
    return inv[selection_label]

# ---------- UI ----------
st.title("Test Deployment Model: College Football Longest Throw Prediction") #🚩change for meaning
st.caption("This is the caption under the title") #🚩change for meaning

st.header("Enter College Football Player's Information") #🚩change for meaning

def num_slider(name, default, lo, hi, step=1):
    r = numeric_ranges.get(name, {})
    lo = int(r.get("min", lo))
    hi = int(r.get("max", hi))
    val = int(r.get("default", default))
    return st.slider(name.replace("_", " ").title(), min_value=lo, max_value=hi, value=val, step=step)

# Numeric features
max_bench = num_slider("max_bench", 225, 100, 405) #🚩change to match numeric features, not including label-encoded categorical columns, change variable name for meaning

st.subheader("Beyond this label: categorical feature dropdown selections") #🚩change for meaning

# Show labels, convert back to codes
throwing_arm_label = st.selectbox("Throwing Arm", list(throwing_arm_map.values())) #🚩change variables to match and for meaning
throwing_arm = label_to_code(throwing_arm_label, throwing_arm_map) #🚩change variables to match and for meaning

# Ordinal: keep label for UX, map to integer for the model
class_label = st.selectbox("Class", class_levels) #🚩change variable name and title to match, change variable name for meaning
Class = class_ord[class_label] #🚩change variable names to match, change variable name for meaning

# ---------- Build raw row ----------
raw_row = {
    # Numeric features
    "max_bench": max_bench, #🚩change variable to match and title for meaning
    # Ordinal numeric
    "class": Class, #🚩change variable to match and title for meaning
    # Categorical codes (as in training)
    "throwing_arm": throwing_arm #🚩change variable to match and title for meaning
}

raw_row = {
    # Numeric features
    "max_bench": max_bench, #🚩change variable to match and title for meaning
    # Ordinal numeric
    "class": Class, #🚩change variable to match and title for meaning
    # Categorical codes (as in training)
    "throwing_arm": throwing_arm #🚩change variable to match and title for meaning
}

raw_df = pd.DataFrame([raw_row])

# ---------- Encode EXACTLY like training ----------
# OHE only these categorical code columns; drop_first=True
ohe_cols = [
    "throwing_arm" #🚩change to match feature column names
]

input_encoded = pd.get_dummies(raw_df, columns=ohe_cols, drop_first=True, dtype=int)

# Make sure all expected training columns exist and in the same order
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[expected_columns]

print(input_encoded)

st.divider()
if st.button("Predict"):
    try:
        pred = model.predict(input_encoded)

        st.subheader("Prediction Result")
        if pred:
            st.success("Prediction: ", round(pred, 2))
        else:
            st.error("Prediction Error")

    except Exception as e:
        st.error(f"Inference failed: {e}")
