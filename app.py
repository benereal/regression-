import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Body Temperature Predictor",
    page_icon="🌡️",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("linear_model.pkl")

model = load_model()

st.title("🌡️ Body Temperature Prediction")
st.markdown(
    """
    Predict oral body temperature using facial infrared thermography measurements.
    """
)

st.divider()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Patient Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

ethnicity = st.sidebar.selectbox(
    "Ethnicity",
    [
        "White",
        "Black",
        "Asian",
        "Hispanic",
        "Other"
    ]
)

age_num = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

# ----------------------------
# Temperature Measurements
# ----------------------------
st.subheader("Thermography Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    T_atm = st.number_input("T_atm", value=25.0)
    Humidity = st.number_input("Humidity", value=50.0)
    Distance = st.number_input("Distance", value=1.0)
    T_offset1 = st.number_input("T_offset1", value=0.0)
    Max1R13_1 = st.number_input("Max1R13_1", value=36.0)
    Max1L13_1 = st.number_input("Max1L13_1", value=36.0)
    aveAllR13_1 = st.number_input("aveAllR13_1", value=36.0)
    aveAllL13_1 = st.number_input("aveAllL13_1", value=36.0)
    T_RC1 = st.number_input("T_RC1", value=36.0)
    T_RC_Dry1 = st.number_input("T_RC_Dry1", value=36.0)
    T_RC_Wet1 = st.number_input("T_RC_Wet1", value=36.0)

with col2:
    T_RC_Max1 = st.number_input("T_RC_Max1", value=36.0)
    T_LC1 = st.number_input("T_LC1", value=36.0)
    T_LC_Dry1 = st.number_input("T_LC_Dry1", value=36.0)
    T_LC_Wet1 = st.number_input("T_LC_Wet1", value=36.0)
    T_LC_Max1 = st.number_input("T_LC_Max1", value=36.0)
    RCC1 = st.number_input("RCC1", value=36.0)
    LCC1 = st.number_input("LCC1", value=36.0)
    canthiMax1 = st.number_input("canthiMax1", value=36.0)
    canthi4Max1 = st.number_input("canthi4Max1", value=36.0)
    T_FHCC1 = st.number_input("T_FHCC1", value=36.0)
    T_FHRC1 = st.number_input("T_FHRC1", value=36.0)

with col3:
    T_FHLC1 = st.number_input("T_FHLC1", value=36.0)
    T_FHBC1 = st.number_input("T_FHBC1", value=36.0)
    T_FHTC1 = st.number_input("T_FHTC1", value=36.0)
    T_FH_Max1 = st.number_input("T_FH_Max1", value=36.0)
    T_FHC_Max1 = st.number_input("T_FHC_Max1", value=36.0)
    T_Max1 = st.number_input("T_Max1", value=36.0)
    T_OR1 = st.number_input("T_OR1", value=36.0)
    T_OR_Max1 = st.number_input("T_OR_Max1", value=36.0)

# Engineered feature used during training
Temp_diff_Wet = T_RC_Wet1 - T_LC_Wet1

st.info(f"Calculated Temp_diff_Wet = {Temp_diff_Wet:.2f}")

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Temperature", use_container_width=True):

    input_df = pd.DataFrame([{
        "Gender": gender,
        "Ethnicity": ethnicity,
        "T_atm": T_atm,
        "Humidity": Humidity,
        "Distance": Distance,
        "T_offset1": T_offset1,
        "Max1R13_1": Max1R13_1,
        "Max1L13_1": Max1L13_1,
        "aveAllR13_1": aveAllR13_1,
        "aveAllL13_1": aveAllL13_1,
        "T_RC1": T_RC1,
        "T_RC_Dry1": T_RC_Dry1,
        "T_RC_Wet1": T_RC_Wet1,
        "T_RC_Max1": T_RC_Max1,
        "T_LC1": T_LC1,
        "T_LC_Dry1": T_LC_Dry1,
        "T_LC_Wet1": T_LC_Wet1,
        "T_LC_Max1": T_LC_Max1,
        "RCC1": RCC1,
        "LCC1": LCC1,
        "canthiMax1": canthiMax1,
        "canthi4Max1": canthi4Max1,
        "T_FHCC1": T_FHCC1,
        "T_FHRC1": T_FHRC1,
        "T_FHLC1": T_FHLC1,
        "T_FHBC1": T_FHBC1,
        "T_FHTC1": T_FHTC1,
        "T_FH_Max1": T_FH_Max1,
        "T_FHC_Max1": T_FHC_Max1,
        "T_Max1": T_Max1,
        "T_OR1": T_OR1,
        "T_OR_Max1": T_OR_Max1,
        "Age_num": age_num,
        "Temp_diff_Wet": Temp_diff_Wet
    }])

    prediction = model.predict(input_df)

    st.success(
        f"Predicted Oral Temperature: {prediction[0][0]:.2f} °C"
    )

    st.subheader("Submitted Data")
    st.dataframe(input_df)