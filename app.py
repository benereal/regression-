import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Body Temperature Predictor",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("linear_model.pkl")

model = load_model()

st.title("Body Temperature Predictor")

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

ethnicity = st.selectbox(
    "Ethnicity",
    [
        "American Indian or Alaskan Native",
        "Asian",
        "Black or African American",
        "Hispanic/Latino",
        "Multiracial",
        "White"
    ]
)

age = st.slider("Age", 1, 100, 30)

st.subheader("Measurements")

values = {}

features = [
    "T_atm",
    "Humidity",
    "Distance",
    "T_offset1",
    "Max1R13_1",
    "Max1L13_1",
    "aveAllR13_1",
    "aveAllL13_1",
    "T_RC1",
    "T_RC_Dry1",
    "T_RC_Wet1",
    "T_RC_Max1",
    "T_LC1",
    "T_LC_Dry1",
    "T_LC_Wet1",
    "T_LC_Max1",
    "RCC1",
    "LCC1",
    "canthiMax1",
    "canthi4Max1",
    "T_FHCC1",
    "T_FHRC1",
    "T_FHLC1",
    "T_FHBC1",
    "T_FHTC1",
    "T_FH_Max1",
    "T_FHC_Max1",
    "T_Max1",
    "T_OR1",
    "T_OR_Max1"
]

cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i % 3]:
        values[feature] = st.number_input(
            feature,
            value=36.0,
            format="%.2f"
        )

if st.button("Predict Temperature"):

    values["Gender"] = gender
    values["Ethnicity"] = ethnicity
    values["Age_num"] = age
    values["Temp_diff_Wet"] = (
        values["T_RC_Wet1"] - values["T_LC_Wet1"]
    )

    try:
        feature_order = model.feature_names_in_

        data = {
            col: values[col]
            for col in feature_order
        }

        df = pd.DataFrame([data])

        prediction = float(model.predict(df)[0])

        st.success("Prediction Complete")

        st.metric(
            "Predicted Temperature",
            f"{prediction:.2f} °C"
        )

        if prediction < 37.5:
            st.success("Normal Temperature")
        elif prediction < 38.5:
            st.warning("Elevated Temperature")
        else:
            st.error("High Temperature")

    except Exception as e:
        st.error(str(e))
