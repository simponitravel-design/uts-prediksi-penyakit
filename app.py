import streamlit as st
import pandas as pd
import joblib
import json

# --- PENGATURAN HALAMAN ---
st.set_page_config(
    page_title="Sistem Prediksi Diagnosis",
    page_icon="🩺",
    layout="wide"
)

# --- FUNGSI UNTUK MEMUAT MODEL ---
# Menggunakan cache agar model tidak perlu dimuat ulang setiap kali ada interaksi
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.joblib')
    return model

@st.cache_data
def load_columns():
    with open('model_columns.json', 'r') as f:
        columns = json.load(f)
    return columns

# Muat model dan kolom
try:
    model = load_model()
    model_columns = load_columns()
except FileNotFoundError:
    st.error("File model atau kolom tidak ditemukan. Pastikan 'random_forest_model.joblib' dan 'model_columns.json' ada di direktori yang sama.")
    st.stop()

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("🩺 Sistem Prediksi Diagnosis Penyakit")
st.write("Pilih gejala yang dialami pasien untuk mendapatkan prediksi diagnosis.")

# Membuat form agar input dikumpulkan sebelum dieksekusi
with st.form("gejala_form"):
    st.header("Pilih Gejala Pasien")

    # Membuat checkbox untuk setiap gejala dalam beberapa kolom
    num_columns = 4
    cols = st.columns(num_columns)
    input_gejala = {}
    
    for i, column_name in enumerate(model_columns):
        with cols[i % num_columns]:
            input_gejala[column_name] = st.checkbox(label=column_name.replace('_', ' ').title(), key=column_name)
    
    # Tombol submit
    submit_button = st.form_submit_button(label="🚀 Prediksi Diagnosis")


# --- LOGIKA PREDIKSI (SETELAH TOMBOL DITEKAN) ---
if submit_button:
    if not any(input_gejala.values()):
        st.warning("Mohon pilih setidaknya satu gejala.")
    else:
        # 1. Buat DataFrame dari input gejala
        input_df = pd.DataFrame([input_gejala])
        
        # 2. Pastikan urutan kolom sesuai dengan saat pelatihan
        input_df = input_df[model_columns]

        # 3. Lakukan prediksi
        prediksi = model.predict(input_df)
        prediksi_proba = model.predict_proba(input_df)

        # 4. Tampilkan hasil
        st.subheader("Hasil Prediksi")
        
        target_cols = ['DIAGNOSA_J', 'DIAGNOSA_R', 'DIAGNOSA_I', 'DIAGNOSA_K', 'DIAGNOSA_LAINNYA']
        ada_prediksi = False

        result_cols = st.columns(len(target_cols))
        for i, target in enumerate(target_cols):
            if prediksi[0][i] == 1:
                probabilitas = prediksi_proba[i][0, 1]
                with result_cols[i]:
                    st.success(f"**{target}**")
                    st.metric(label="Tingkat Keyakinan", value=f"{probabilitas:.1%}")
                ada_prediksi = True
        
        if not ada_prediksi:
            st.info("Berdasarkan gejala yang dipilih, tidak ada kategori diagnosis spesifik yang terprediksi secara signifikan.")