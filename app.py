"""
Ames Housing Price Prediction
Aplikasi inferensi harga properti berbasis Streamlit.
Model: Gradient Boosting Regressor (Tuned)
Serialisasi: Joblib dan Pickle
"""

import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN STREAMLIT
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Ames Housing Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS Bersih dan Profesional
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: #0f172a;
        border-radius: 12px;
        padding: 24px 28px;
        color: #ffffff;
        margin-bottom: 24px;
        border: 1px solid #1e293b;
    }
    
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
        color: #f8fafc;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.4;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 10px;
        background: #1e293b;
        color: #cbd5e1;
        border: 1px solid #334155;
    }

    .result-card {
        background: #0f172a;
        border: 1px solid #38bdf8;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #ffffff;
        margin: 16px 0;
    }
    
    .price-usd {
        font-size: 2.8rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 4px 0;
    }
    
    .price-idr {
        font-size: 1.25rem;
        font-weight: 600;
        color: #34d399;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. PEMUATAN MODEL & ARTIFACT (CACHED)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat pipeline model...")
def load_pipeline():
    """Memuat bundle pipeline dari file joblib atau pickle."""
    joblib_path = "house_price_pipeline.joblib"
    pickle_path = "house_price_pipeline.pkl"
    
    bundle = None
    if os.path.exists(joblib_path):
        try:
            bundle = joblib.load(joblib_path)
            bundle["_loaded_from"] = joblib_path
        except Exception:
            pass
            
    if bundle is None and os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                bundle = pickle.load(f)
            bundle["_loaded_from"] = pickle_path
        except Exception:
            pass
            
    return bundle

bundle = load_pipeline()

# ------------------------------------------------------------------
# 3. HEADER
# ------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Ames Housing Price Prediction</div>
    <div class="hero-subtitle">
        Sistem inferensi nilai jual properti menggunakan algoritma Gradient Boosting Regressor terkalibrasi.<br>
        Pipeline mencakup penanganan nilai hilang, eliminasi outlier (IQR), standardisasi Z-Score, dan Label Encoding.
    </div>
    <div>
        <span class="badge-pill">Test R2 = 0.8915</span>
        <span class="badge-pill">Test MAE = USD 11,870.68</span>
        <span class="badge-pill">Test MAPE = 7.73%</span>
        <span class="badge-pill">Format Serialisasi: Joblib & Pickle</span>
    </div>
</div>
""", unsafe_allow_html=True)

if bundle is None:
    st.error("Artifact pipeline tidak ditemukan (house_price_pipeline.joblib atau house_price_pipeline.pkl). Silakan jalankan 'python train_and_export.py' terlebih dahulu.")
    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
scaler_idx = bundle["scaler_feature_indices"]
encoders = bundle["label_encoders"]
sp_mean = bundle["sp_mean"]
sp_scale = bundle["sp_scale"]
feature_names = bundle["feature_names"]
defaults_num = bundle["defaults_numeric"]
defaults_cat = bundle["defaults_categorical"]
cat_options = bundle["categorical_options"]
top_features_list = bundle["top_features"]
metrics = bundle["metrics"]

# ------------------------------------------------------------------
# 4. SIDEBAR - PRESET & PENGATURAN KURS
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Pengaturan dan Preset")
    
    preset_choice = st.selectbox(
        "Pilih Profil Preset:",
        [
            "Kustom (Input Manual)",
            "Tipe A (Luas dan Kualitas Tinggi)",
            "Tipe B (Menengah / Standar Keluarga)",
            "Tipe C (Kompak / Minimalis)"
        ],
        index=0
    )
    
    if preset_choice == "Tipe A (Luas dan Kualitas Tinggi)":
        p_qual, p_cond, p_year, p_remod = 9, 7, 2010, 2015
        p_grliv, p_bsmt, p_garage_cars, p_garage_area = 2450, 1450, 3, 850
        p_bath, p_bed = 3, 4
        p_neigh, p_exter, p_kitchen, p_air = "NridgHt", "Gd", "Gd", "Y"
    elif preset_choice == "Tipe B (Menengah / Standar Keluarga)":
        p_qual, p_cond, p_year, p_remod = 7, 6, 2002, 2005
        p_grliv, p_bsmt, p_garage_cars, p_garage_area = 1750, 950, 2, 480
        p_bath, p_bed = 2, 3
        p_neigh, p_exter, p_kitchen, p_air = "CollgCr", "TA", "TA", "Y"
    elif preset_choice == "Tipe C (Kompak / Minimalis)":
        p_qual, p_cond, p_year, p_remod = 5, 5, 1978, 1985
        p_grliv, p_bsmt, p_garage_cars, p_garage_area = 980, 650, 1, 280
        p_bath, p_bed = 1, 2
        p_neigh, p_exter, p_kitchen, p_air = "Edwards", "TA", "TA", "Y"
    else:
        p_qual = int(defaults_num.get("OverallQual", 6))
        p_cond = int(defaults_num.get("OverallCond", 5))
        p_year = int(defaults_num.get("YearBuilt", 1973))
        p_remod = int(defaults_num.get("YearRemodAdd", 1984))
        p_grliv = int(defaults_num.get("GrLivArea", 1464))
        p_bsmt = int(defaults_num.get("TotalBsmtSF", 991))
        p_garage_cars = int(defaults_num.get("GarageCars", 2))
        p_garage_area = int(defaults_num.get("GarageArea", 480))
        p_bath = int(defaults_num.get("FullBath", 2))
        p_bed = int(defaults_num.get("BedroomAbvGr", 3))
        p_neigh = defaults_cat.get("Neighborhood", "NAmes")
        p_exter = defaults_cat.get("ExterQual", "TA")
        p_kitchen = defaults_cat.get("KitchenQual", "TA")
        p_air = defaults_cat.get("CentralAir", "Y")

    st.markdown("---")
    st.subheader("Konversi Mata Uang")
    kurs_usd_idr = st.number_input(
        "Kurs USD ke IDR:",
        min_value=10000,
        max_value=25000,
        value=16000,
        step=500
    )

    st.markdown("---")
    st.caption(f"Artifact loaded: {bundle.get('_loaded_from', 'house_price_pipeline.joblib')}")
    st.caption("Model: Gradient Boosting Regressor (150 trees, max_depth=3)")

# ------------------------------------------------------------------
# 5. TAB NAVIGASI UTAMA
# ------------------------------------------------------------------
tab_prediksi, tab_analisis, tab_info = st.tabs([
    "Prediksi Harga Rumah",
    "Analisis Fitur",
    "Spesifikasi Model"
])

# ==================================================================
# TAB 1: FORMULIR PREDIKSI
# ==================================================================
with tab_prediksi:
    st.subheader("Parameter Properti")
    st.write("Masukkan nilai parameter spesifikasi bangunan di bawah ini:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Kualitas Material dan Waktu Konstruksi**")
        input_qual = st.slider("Kualitas Material Keseluruhan (OverallQual, skala 1-10):", 1, 10, p_qual)
        input_cond = st.slider("Kondisi Fisik Bangunan (OverallCond, skala 1-10):", 1, 10, p_cond)
        
        c_yr1, c_yr2 = st.columns(2)
        with c_yr1:
            input_year = st.number_input("Tahun Konstruksi (YearBuilt):", min_value=1870, max_value=2015, value=p_year)
        with c_yr2:
            input_remod = st.number_input("Tahun Renovasi (YearRemodAdd):", min_value=1950, max_value=2015, value=p_remod)

        st.markdown("**Lokasi dan Kualitas Komponen**")
        neigh_options = cat_options.get("Neighborhood", ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt"])
        input_neigh = st.selectbox(
            "Lingkungan / Kawasan (Neighborhood):",
            neigh_options,
            index=neigh_options.index(p_neigh) if p_neigh in neigh_options else 0
        )

        c_eq, c_kq = st.columns(2)
        with c_eq:
            exter_opts = cat_options.get("ExterQual", ["TA", "Gd", "Ex", "Fa"])
            input_exter = st.selectbox("Kualitas Eksterior (ExterQual):", exter_opts, index=exter_opts.index(p_exter) if p_exter in exter_opts else 0)
        with c_kq:
            kitchen_opts = cat_options.get("KitchenQual", ["TA", "Gd", "Ex", "Fa"])
            input_kitchen = st.selectbox("Kualitas Dapur (KitchenQual):", kitchen_opts, index=kitchen_opts.index(p_kitchen) if p_kitchen in kitchen_opts else 0)

        air_opts = cat_options.get("CentralAir", ["Y", "N"])
        input_air = st.radio("Pendingin Udara Sentral (CentralAir):", air_opts, index=air_opts.index(p_air) if p_air in air_opts else 0, horizontal=True)

    with col2:
        st.markdown("**Dimensi dan Luas Bangunan**")
        input_grliv = st.slider("Luas Area Hunian di Atas Tanah (GrLivArea, sqft):", 300, 3000, p_grliv)
        input_bsmt = st.slider("Luas Area Bawah Tanah (TotalBsmtSF, sqft):", 0, 2500, p_bsmt)

        c_gar1, c_gar2 = st.columns(2)
        with c_gar1:
            input_garage_cars = st.slider("Kapasitas Garasi Mobil (GarageCars):", 0, 4, p_garage_cars)
        with c_gar2:
            input_garage_area = st.number_input("Luas Garasi (GarageArea, sqft):", min_value=0, max_value=1400, value=p_garage_area)

        st.markdown("**Fasilitas Ruangan**")
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            input_bath = st.slider("Kamar Mandi Lengkap (FullBath):", 0, 4, p_bath)
        with c_b2:
            input_bed = st.slider("Kamar Tidur di Atas Tanah (BedroomAbvGr):", 0, 6, p_bed)

        input_fireplaces = st.slider("Jumlah Perapian (Fireplaces):", 0, 3, int(defaults_num.get("Fireplaces", 1)))

    with st.expander("Fitur Tambahan dan Detail Sekunder"):
        st.caption("Fitur berikut diisi secara default berdasarkan nilai median dan modus data latihan.")
        c_sub1, c_sub2, c_sub3 = st.columns(3)
        with c_sub1:
            input_lotarea = st.number_input("Luas Lahan (LotArea, sqft):", min_value=1000, max_value=50000, value=int(defaults_num.get("LotArea", 9500)))
            bldg_opts = cat_options.get("BldgType", ["1Fam", "TwnhsE", "Duplex", "Twnhs", "2fmCon"])
            input_bldg = st.selectbox("Tipe Bangunan (BldgType):", bldg_opts)
        with c_sub2:
            style_opts = cat_options.get("HouseStyle", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"])
            input_style = st.selectbox("Model Arsitektur (HouseStyle):", style_opts)
            input_totrms = st.slider("Total Ruangan di Atas Tanah (TotRmsAbvGrd):", 2, 14, int(defaults_num.get("TotRmsAbvGrd", 6)))
        with c_sub3:
            found_opts = cat_options.get("Foundation", ["PConc", "CBlock", "BrkTil", "Slab"])
            input_foundation = st.selectbox("Tipe Pondasi (Foundation):", found_opts)
            heat_opts = cat_options.get("HeatingQC", ["Ex", "TA", "Gd", "Fa"])
            input_heatingqc = st.selectbox("Kualitas Pemanas (HeatingQC):", heat_opts)

    st.markdown("---")
    
    # --------------------------------------------------------------
    # EKSEKUSI PREDIKSI
    # --------------------------------------------------------------
    btn_predict = st.button("Hitung Estimasi Harga", type="primary", use_container_width=True)

    current_features = {}
    for col in feature_names:
        if col in defaults_num:
            current_features[col] = float(defaults_num[col])
        elif col in defaults_cat:
            current_features[col] = str(defaults_cat[col])
        else:
            current_features[col] = 0.0

    current_features["OverallQual"] = float(input_qual)
    current_features["OverallCond"] = float(input_cond)
    current_features["YearBuilt"] = float(input_year)
    current_features["YearRemodAdd"] = float(input_remod)
    current_features["Neighborhood"] = str(input_neigh)
    current_features["ExterQual"] = str(input_exter)
    current_features["KitchenQual"] = str(input_kitchen)
    current_features["CentralAir"] = str(input_air)
    current_features["GrLivArea"] = float(input_grliv)
    current_features["TotalBsmtSF"] = float(input_bsmt)
    current_features["GarageCars"] = float(input_garage_cars)
    current_features["GarageArea"] = float(input_garage_area)
    current_features["FullBath"] = float(input_bath)
    current_features["BedroomAbvGr"] = float(input_bed)
    current_features["Fireplaces"] = float(input_fireplaces)
    current_features["LotArea"] = float(input_lotarea)
    current_features["BldgType"] = str(input_bldg)
    current_features["HouseStyle"] = str(input_style)
    current_features["TotRmsAbvGrd"] = float(input_totrms)
    current_features["Foundation"] = str(input_foundation)
    current_features["HeatingQC"] = str(input_heatingqc)

    processed_row = {}
    for col in feature_names:
        if col in defaults_num and col in scaler_idx:
            val = float(current_features[col])
            idx = scaler_idx[col]
            val_scaled = (val - scaler.mean_[idx]) / scaler.scale_[idx]
            processed_row[col] = val_scaled
        elif col in defaults_cat and col in encoders:
            val_str = str(current_features[col])
            le = encoders[col]
            if val_str in le.classes_:
                processed_row[col] = int(le.transform([val_str])[0])
            else:
                processed_row[col] = 0
        else:
            processed_row[col] = 0.0

    df_input = pd.DataFrame([processed_row])[feature_names]
    pred_standardized = float(model.predict(df_input)[0])
    predicted_usd = (pred_standardized * sp_scale) + sp_mean
    predicted_idr = predicted_usd * kurs_usd_idr

    mae_margin = float(metrics.get("mae_usd", 11870.68))
    low_usd = max(0, predicted_usd - mae_margin)
    high_usd = predicted_usd + mae_margin
    low_idr = low_usd * kurs_usd_idr
    high_idr = high_usd * kurs_usd_idr

    if predicted_usd >= 280000:
        tier_label = "Kategori: Properti Mewah"
        tier_color = "#38bdf8"
    elif predicted_usd >= 180000:
        tier_label = "Kategori: Properti Menengah Atas"
        tier_color = "#34d399"
    elif predicted_usd >= 130000:
        tier_label = "Kategori: Properti Menengah Standar"
        tier_color = "#fbbf24"
    else:
        tier_label = "Kategori: Properti Ekonomis"
        tier_color = "#f87171"

    st.markdown(f"""
    <div class="result-card">
        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; font-weight: 600;">
            ESTIMASI NILAI JUAL RUMAH
        </div>
        <div class="price-usd">${predicted_usd:,.2f}</div>
        <div class="price-idr">Rp {predicted_idr:,.0f}</div>
        <div style="margin-top: 10px; font-size: 0.9rem; color: #cbd5e1;">
            <b>Rentang Estimasi (Toleransi Test MAE ${mae_margin:,.2f}):</b><br>
            ${low_usd:,.2f} — ${high_usd:,.2f}<br>
            <span style="color: #94a3b8;">(Rp {low_idr:,.0f} — Rp {high_idr:,.0f})</span>
        </div>
        <div style="margin-top: 14px;">
            <span style="border: 1px solid {tier_color}; color: {tier_color}; padding: 4px 14px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;">
                {tier_label}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 2: ANALISIS KEPENTINGAN FITUR
# ==================================================================
with tab_analisis:
    st.subheader("Analisis Kepentingan Fitur (Feature Importance)")
    st.write("Kontribusi relatif 15 fitur utama terhadap prediksi harga rumah pada model Gradient Boosting:")

    df_imp = pd.DataFrame(top_features_list)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color="#38bdf8", height=0.65)
    ax.bar_label(bars, fmt="%.3f", padding=5, color="white", fontsize=9)

    ax.set_title("15 Fitur Paling Berpengaruh Terhadap SalePrice", color="white", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Bobot Kepentingan Relatif", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.3, color="#64748b")
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown("""
    **Catatan Analisis:**
    1. **OverallQual:** Kualitas material dan pengerjaan konstruksi secara konsisten menjadi prediktor dengan bobot tertinggi.
    2. **GrLivArea:** Luas area tempat tinggal di atas tanah memiliki korelasi linier positif terkuat terhadap harga.
    3. **TotalBsmtSF:** Luas ruang bawah tanah memberikan kontribusi signifikan terhadap estimasi nilai properti.
    4. **GarageCars dan GarageArea:** Kapasitas fasilitas garasi kendaraan menjadi indikator kelas segmen hunian.
    5. **YearBuilt:** Usia bangunan merefleksikan tingkat depresiasi fisik dan standar arsitektur.
    """)

# ==================================================================
# TAB 3: SPESIFIKASI MODEL
# ==================================================================
with tab_info:
    st.subheader("Spesifikasi Model dan Evaluasi Kinerja")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test R2", f"{metrics.get('r2', 0.8915):.4f}")
    with col_m2:
        st.metric("Test RMSE", f"${metrics.get('rmse_usd', 16299.45):,.2f}")
    with col_m3:
        st.metric("Test MAE", f"${metrics.get('mae_usd', 11870.68):,.2f}")
    with col_m4:
        st.metric("Test MAPE", f"{metrics.get('mape', 7.73):.2f}%")

    st.markdown("---")
    st.subheader("Pemuatan Model (Joblib dan Pickle)")
    st.write("Contoh sintaks pemuatan file artifact serialisasi pada lingkungan Python:")

    code_snippet = """# 1. Pemuatan menggunakan Joblib
import joblib

# Muat estimator
model = joblib.load('model.joblib')
# Atau muat pipeline bundle lengkap
pipeline = joblib.load('house_price_pipeline.joblib')

# 2. Pemuatan menggunakan Pickle
import pickle

with open('model.pkl', 'rb') as f:
    model_pickle = pickle.load(f)

with open('house_price_pipeline.pkl', 'rb') as f:
    pipeline_pickle = pickle.load(f)

# Prediksi menggunakan pipeline bundle
scaled_features = ...  # Matriks 75 fitur yang telah distandardisasi
y_norm = pipeline['model'].predict(scaled_features)
sale_price_usd = y_norm * pipeline['sp_scale'] + pipeline['sp_mean']
print(f"Hasil Prediksi: ${sale_price_usd[0]:,.2f}")
"""
    st.code(code_snippet, language="python")

    st.markdown("---")
    st.subheader("Tahapan Pipeline Preprocessing")
    st.markdown("""
    1. **Imputasi Missing Values:** Median untuk fitur numerik, modus untuk fitur kategorikal.
    2. **Eliminasi Fitur Kosong (>75%):** Kolom Alley, PoolQC, Fence, dan MiscFeature dieliminasi.
    3. **Pembersihan Outlier (IQR):** Baris data di luar batas 1.5 * IQR pada kolom numerik dihapus (557 baris tersisa).
    4. **Standardisasi:** StandardScaler diterapkan pada seluruh fitur numerik.
    5. **Encoding Fitur Kategorikal:** LabelEncoder diterapkan pada 39 fitur kategorik.
    6. **Split Latih dan Uji:** Pembagian data dengan proporsi 80% Latih dan 20% Uji (random_state=1).
    """)

if __name__ == "__main__":
    pass
