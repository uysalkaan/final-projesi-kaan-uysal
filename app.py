import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_delivery_model.pkl")

st.title("Smart Delivery Risk Predictor")
st.write("Online siparişler için teslimat gecikme riski tahmin sistemi")

distance = st.number_input("Mesafe (km)", min_value=0.1, max_value=30.0, value=10.0)
weather = st.selectbox("Hava Durumu", ["Clear", "Rainy", "Foggy", "Windy", "Snowy"])
traffic = st.selectbox("Trafik Yoğunluğu", ["Low", "Medium", "High"])
time_of_day = st.selectbox("Günün Zamanı", ["Morning", "Afternoon", "Evening", "Night"])
vehicle = st.selectbox("Araç Tipi", ["Bike", "Scooter", "Car"])
prep_time = st.number_input("Hazırlık Süresi (dk)", min_value=1, max_value=60, value=15)
experience = st.number_input("Kurye Deneyimi (yıl)", min_value=0.0, max_value=20.0, value=3.0)

input_data = pd.DataFrame({
    "Distance_km": [distance],
    "Weather": [weather],
    "Traffic_Level": [traffic],
    "Time_of_Day": [time_of_day],
    "Vehicle_Type": [vehicle],
    "Preparation_Time_min": [prep_time],
    "Courier_Experience_yrs": [experience]
})

if st.button("Teslimat Riskini Tahmin Et"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Tahmin Sonucu")

    if prediction == 1:
        st.error(f"Yüksek Teslimat Gecikme Riski: %{probability * 100:.2f}")
    else:
        st.success(f"Düşük Teslimat Gecikme Riski: %{probability * 100:.2f}")

    st.write("Girilen Bilgiler:")
    st.dataframe(input_data)