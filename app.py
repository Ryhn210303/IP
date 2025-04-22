import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("destinasi-wisata-indonesia.csv")

df = load_data()

# Ambil kolom yang dibutuhkan
data = df[['Place_Name', 'Category', 'City', 'Price', 'Rating']].copy()

# Preprocessing
encoder_kat = OneHotEncoder(sparse_output=False)
encoder_kota = OneHotEncoder(sparse_output=False)
kategori_encoded = encoder_kat.fit_transform(data[['Category']])
kota_encoded = encoder_kota.fit_transform(data[['City']])

scaler = StandardScaler()
numerik_scaled = scaler.fit_transform(data[['Price', 'Rating']])

fitur = np.hstack([kategori_encoded, kota_encoded, numerik_scaled])

# Train model K-NN
model_knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
model_knn.fit(fitur)

# UI Streamlit
st.title("🎯 Sistem Rekomendasi Destinasi Wisata")
st.write("Berdasarkan kategori wisata dan fitur kesamaan.")

# Input kategori
kategori_user = st.selectbox("Pilih kategori wisata:", sorted(data['Category'].unique()))

if st.button("Cari Rekomendasi"):
    # Cari baris referensi dari input user
    if kategori_user in data['Category'].values:
        idx = data[data['Category'] == kategori_user].index[0]
        user_fitur = fitur[idx].reshape(1, -1)

        # Cari rekomendasi
        distances, indices = model_knn.kneighbors(user_fitur)

        rekomendasi = data.iloc[indices[0]].reset_index(drop=True)
        st.success("Berikut rekomendasi wisata yang cocok untuk Anda:")
        st.dataframe(rekomendasi)
    else:
        st.warning("Kategori tidak ditemukan.")
