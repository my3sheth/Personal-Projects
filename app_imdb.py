import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# Load the model and preprocessor
model = load_model("model.h5")

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.title("🎬 IMDB Movie Revenue Predictor")

# Input fields
year = st.number_input("Year of Release", min_value=1900, max_value=2100, value=2020)
certificate = st.selectbox("Certificate", ['U', 'UA', 'A', 'PG-13', 'R', 'Not Rated'])
runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=300, value=120)
genre = st.selectbox("Genre", ['Action', 'Drama', 'Comedy', 'Thriller', 'Other'])
rating = st.slider("Rating (0.0 to 10.0)", min_value=0.0, max_value=10.0, value=7.5)
meta_score = st.number_input("Meta Score", min_value=0, max_value=100, value=60)
director = st.text_input("Director", value="Unknown")
star_1 = st.text_input("Star 1", value="Actor A")
star_2 = st.text_input("Star 2", value="Actor B")
star_3 = st.text_input("Star 3", value="Actor C")
star_4 = st.text_input("Star 4", value="Actor D")
votes = st.number_input("Total Votes", min_value=0, value=100000)

if st.button("Predict Revenue"):
    input_df = pd.DataFrame([{
        'Year_of_Release': year,
        'Certificate': certificate,
        'Runtime': runtime,
        'Genre': genre,
        'Rating': rating,
        'Meta_Score': meta_score,
        'Director': director,
        'Star_1': star_1,
        'Star_2': star_2,
        'Star_3': star_3,
        'Star_4': star_4,
        'Total_Votes': votes
    }])

    input_scaled = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    st.success(f"💰 Predicted Revenue: ${prediction[0][0]:,.2f}")
