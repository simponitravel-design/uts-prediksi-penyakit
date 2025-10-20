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

# --- FUNGSI & DATA UNTUK PEMROSESAN TEKS ---

@st.cache_data
def load_symptom_dictionary():
    """Memuat dan memproses kamus gejala dari file Excel."""
    try:
        file_path = 'KANDIDAT_GEJALA_UNTUK_DIKURASI.xlsx'
        df_kurasi = pd.read_excel(file_path)

        # Ganti nama kolom agar lebih mudah digunakan dalam kode
        df_kurasi = df_kurasi.rename(columns={
            'Kandidat dari Mesin (Sudah di-Stem)': 'kandidat',
            'Gejala Standar (MOHON DIISI MANUAL)': 'standar'
        })
        
        # Hapus baris di mana gejala standar tidak diisi
        df_kurasi.dropna(subset=['standar'], inplace=True)
        
        # Urutkan dari kandidat terpanjang agar deteksi lebih akurat
        df_kurasi['len'] = df_kurasi['kandidat'].astype(str).str.len()
        df_kurasi = df_kurasi.sort_values('len', ascending=False).drop('len', axis=1)
        
        smart_dictionary = dict(zip(df_kurasi.kandidat, df_kurasi.standar))
        return smart_dictionary
    except FileNotFoundError:
        st.error(f"File kamus gejala '{file_path}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan app.py.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file Excel: {e}")
        return None


def extract_symptoms_from_text(text, dictionary):
    """Mengekstrak gejala standar dari teks keluhan pasien."""
    if dictionary is None:
        return []
    found_symptoms = set()
    processed_text = text.lower()
    for kandidat, gejala_standar in dictionary.items():
        # Pastikan kandidat adalah string sebelum dicari
        if isinstance(kandidat, str) and kandidat in processed_text:
            found_symptoms.add(gejala_standar)
    return list(found_symptoms)

# --- FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.joblib')
    return model

@st.cache_data
def load_columns():
    with open('model_columns.json', 'r') as f:
        columns = json.load(f)
    return columns

# Muat model, kolom, dan kamus gejala
try:
    model = load_model()
    model_columns = load_columns()
    symptom_dictionary = load_symptom_dictionary()
except FileNotFoundError:
    st.error("File model atau kolom tidak ditemukan. Pastikan 'random_forest_model.joblib' dan 'model_columns.json' ada di direktori yang sama.")
    st.stop()

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("🩺 Sistem Prediksi Diagnosis Penyakit")
st.write("Silakan masukkan keluhan atau gejala yang dialami pasien dalam bentuk kalimat, lalu tekan tombol 'Prediksi Diagnosis'.")

with st.form("gejala_form"):
    st.header("Masukkan Keluhan Pasien")
    user_input_text = st.text_area(
        label="Contoh: 'pasien mengeluh pusing dan mual sejak kemarin, disertai demam ringan'",
        height=100
    )
    submit_button = st.form_submit_button(label="🚀 Prediksi Diagnosis")

# --- LOGIKA PREDIKSI & TAMPILAN HASIL ---
if submit_button:
    if not user_input_text.strip():
        st.warning("Mohon masukkan teks keluhan pasien terlebih dahulu.")
    elif symptom_dictionary is None:
        # Pesan error sudah ditampilkan di dalam fungsi load
        pass
    else:
        # --- FASE 1: Proses Data di Belakang Layar ---
        gejala_terdeteksi = extract_symptoms_from_text(user_input_text, symptom_dictionary)
        prediksi = None
        prediksi_proba = None
        
        if gejala_terdeteksi:
            input_data = {col: [0] for col in model_columns}
            for gejala in gejala_terdeteksi:
                if gejala in input_data:
                    input_data[gejala] = [1]
            input_df = pd.DataFrame(input_data)[model_columns]
            
            try:
                prediksi = model.predict(input_df)
                prediksi_proba = model.predict_proba(input_df)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

        st.divider()

        # --- FASE 2: Tampilkan Hasil dalam Dua Kolom ---
        col1, col2 = st.columns([1, 2])

        # --- KOLOM KIRI: Gejala Teridentifikasi ---
        with col1:
            st.subheader("📝 Gejala Teridentifikasi")
            if not gejala_terdeteksi:
                st.info("Tidak ada gejala yang dikenali dari teks yang Anda masukkan.")
            else:
                for gejala in sorted(gejala_terdeteksi):
                    st.markdown(f"- `{gejala.replace('_', ' ').title()}`")

        # --- KOLOM KANAN: Hasil Prediksi ---
        with col2:
            st.subheader("📈 Hasil Prediksi Diagnosis")
            if prediksi is not None and prediksi_proba is not None:
                target_cols = ['DIAGNOSA_J', 'DIAGNOSA_R', 'DIAGNOSA_I', 'DIAGNOSA_K', 'DIAGNOSA_LAINNYA']
                
                # [PERUBAHAN] Kamus untuk deskripsi diagnosis
                diagnosis_descriptions = {
                    'DIAGNOSA_J': "Diagnosis terkait sistem pernapasan.",
                    'DIAGNOSA_R': "Diagnosis terkait gejala tidak spesifik atau umum.",
                    'DIAGNOSA_I': "Diagnosis terkait sistem peredaran darah.",
                    'DIAGNOSA_K': "Diagnosis terkait sistem pencernaan.",
                    'DIAGNOSA_LAINNYA': "Diagnosis lain di luar kategori J, R, I, K."
                }
                
                ada_prediksi = False

                for i, target in enumerate(target_cols):
                    if prediksi[0][i] == 1:
                        probabilitas = prediksi_proba[i][0, 1]
                        deskripsi = diagnosis_descriptions.get(target, "Deskripsi tidak tersedia.")
                        
                        st.success(f"**{target.replace('_', ' ').title()}**")
                        
                        # [PERUBAHAN] Menambahkan deskripsi
                        st.caption(deskripsi) 
                        
                        st.metric(label="Tingkat Keyakinan", value=f"{probabilitas:.1%}")
                        st.write("") 
                        ada_prediksi = True
                
                if not ada_prediksi:
                    st.info("Berdasarkan gejala yang teridentifikasi, tidak ada diagnosis spesifik yang terprediksi secara signifikan.")
            elif gejala_terdeteksi:
                st.warning("Prediksi tidak dapat dilakukan karena terjadi kesalahan internal.")
            else:
                st.info("Prediksi tidak dapat ditampilkan karena tidak ada gejala yang teridentifikasi.")