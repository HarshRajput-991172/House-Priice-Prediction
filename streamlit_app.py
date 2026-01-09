import streamlit as st
import joblib
import pandas as pd

# Load models
model_lr = joblib.load("ML_Models/model.joblib")
scaler = joblib.load("ML_Models/scaler.joblib")

st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction")

st.write("Enter the house details below:")

# ---- USER INPUTS ----
area = st.number_input("Area (sq ft)", min_value=100)
bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, step=1)
stories = st.number_input("Stories", min_value=1, step=1)
parking = st.number_input("Parking Spaces", min_value=0, step=1)

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnishing_status = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# ---- FURNISHING LOGIC ----
if furnishing_status == "furnished":
    furnishing_semi = "no"
    furnishing_unfurnished = "no"
elif furnishing_status == "semi-furnished":
    furnishing_semi = "yes"
    furnishing_unfurnished = "no"
else:
    furnishing_semi = "no"
    furnishing_unfurnished = "yes"

# ---- PREDICTION ----
if st.button("Predict Price"):

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus_semi-furnished": furnishing_semi,
        "furnishingstatus_unfurnished": furnishing_unfurnished
    }

    df = pd.DataFrame(input_data, index=[0])

    yes_no_cols = [
        "mainroad", "guestroom", "basement", "hotwaterheating",
        "airconditioning", "prefarea",
        "furnishingstatus_semi-furnished",
        "furnishingstatus_unfurnished"
    ]

    mapping = {"yes": 1, "no": 0}
    for col in yes_no_cols:
        df[col] = df[col].map(mapping)

    num_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
    df[num_cols] = scaler.transform(df[num_cols])

    prediction = model_lr.predict(df)[0]

    st.success(f"💰 Estimated House Price: ₹ {round(prediction, 2):,.2f}")
