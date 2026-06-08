from pathlib import Path

app_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Human Temperature Predictor",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("linear_model.pkl")

model = load_model()

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.metric-card {
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #dcdcdc;
}
</style>
""", unsafe_allow_html=True)

st.title("Human Temperature Prediction System")
st.caption("Infrared Thermography Based Body Temperature Estimator")

with st.sidebar:
    st.header("Subject Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox(
        "Ethnicity",
        [
            "White", "Black", "Asian", "Hispanic",
            "Middle Eastern", "Native American", "Other"
        ]
    )
    age_num = st.slider("Age", 1, 90, 25)

st.subheader("Environmental Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    t_atm = st.number_input("Atmospheric Temp (°C)", value=24.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    distance = st.number_input("Distance", value=1.0)

with col2:
    t_offset1 = st.number_input("T_offset1", value=0.0)
    max1r13 = st.number_input("Max1R13_1", value=36.0)
    max1l13 = st.number_input("Max1L13_1", value=36.0)

with col3:
    ave_r = st.number_input("aveAllR13_1", value=36.0)
    ave_l = st.number_input("aveAllL13_1", value=36.0)

st.subheader("Thermal Measurements")

feature_names = [
    'T_RC1','T_RC_Dry1','T_RC_Wet1','T_RC_Max1',
    'T_LC1','T_LC_Dry1','T_LC_Wet1','T_LC_Max1',
    'RCC1','LCC1','canthiMax1','canthi4Max1',
    'T_FHCC1','T_FHRC1','T_FHLC1','T_FHBC1',
    'T_FHTC1','T_FH_Max1','T_FHC_Max1','T_Max1',
    'T_OR1','T_OR_Max1'
]

values = {}
cols = st.columns(4)

for i, feature in enumerate(feature_names):
    with cols[i % 4]:
        values[feature] = st.number_input(feature, value=36.0, key=feature)

temp_diff_wet = values["T_RC_Wet1"] - values["T_LC_Wet1"]

if st.button("Predict Temperature", use_container_width=True):

    data = {
        "Gender": gender,
        "Ethnicity": ethnicity,
        "T_atm": t_atm,
        "Humidity": humidity,
        "Distance": distance,
        "T_offset1": t_offset1,
        "Max1R13_1": max1r13,
        "Max1L13_1": max1l13,
        "aveAllR13_1": ave_r,
        "aveAllL13_1": ave_l,
        **values,
        "Age_num": age_num,
        "Temp_diff_Wet": temp_diff_wet
    }

    df = pd.DataFrame([data])

    try:
        prediction = model.predict(df)

        if isinstance(prediction, np.ndarray):
            pred_value = float(np.ravel(prediction)[0])
        else:
            pred_value = float(prediction)

        st.success("Prediction Complete")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Predicted Temperature", f"{pred_value:.2f} °C")

        with c2:
            status = "Normal" if pred_value < 37.5 else "Elevated"
            st.metric("Status", status)

        with c3:
            risk = "Low" if pred_value < 37.5 else "Monitor"
            st.metric("Assessment", risk)

        with st.expander("View Submitted Features"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built with Streamlit and Scikit-Learn")
'''
Path("/mnt/data/app.py").write_text(app_code)
print("/mnt/data/app.py")
