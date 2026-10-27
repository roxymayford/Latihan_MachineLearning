"""
====================================================================
🏡 House Price Prediction AI - Ames Housing Dataset
Aplikasi Web Frontend Interaktif Berbasis Streamlit
Model: Gradient Boosting Regressor (Tuned) - R^2 = 89.15%
Format Serialisasi: Joblib (.joblib) & Pickle (.pkl)
====================================================================
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
# 1. KONFIGURASI HALAMAN & TEMA STREAMLIT
# ------------------------------------------------------------------
st.set_page_config(
    page_title="House Price AI - Prediksi Harga Rumah",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS untuk Antarmuka Modern & Premium
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border-radius: 16px;
        padding: 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        font-weight: 400;
        line-height: 1.5;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 12px;
    }
    
    .badge-primary { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .badge-success { background: rgba(52, 211, 153, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.4); }
    .badge-warning { background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.4); }

    .result-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 2px solid #38bdf8;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(56, 189, 248, 0.15);
        margin: 16px 0;
    }
    
    .price-usd {
        font-size: 3rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 4px 0;
    }
    
    .price-idr {
        font-size: 1.35rem;
        font-weight: 600;
        color: #34d399;
        margin-bottom: 12px;
    }
    
    .info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. PEMUATAN MODEL & ARTIFACT (CACHED)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model Machine Learning...")
def load_pipeline():
    """Mencari dan memuat pipeline dari file joblib atau pickle."""
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
# 3. HEADER & HERO SECTION
# ------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏡 Ames Housing Price Prediction AI</div>
    <div class="hero-subtitle">
        Sistem estimasi nilai jual rumah cerdas berbasis algoritma <b>Gradient Boosting Regressor (Tuned)</b>.<br>
        Dibangun melalui pipeline terstandardisasi: pembersihan missing value, eliminasi outlier (IQR), Z-Score scaling, dan Label Encoding.
    </div>
    <div>
        <span class="badge-pill badge-primary">🎯 Test R² = 89.15%</span>
        <span class="badge-pill badge-success">📉 Test MAE = ±$11,870 (7.73%)</span>
        <span class="badge-pill badge-warning">📦 Serialisasi: Joblib & Pickle (.joblib, .pkl)</span>
    </div>
</div>
""", unsafe_allow_html=True)

if bundle is None:
    st.error("⚠️ Artifact pipeline model tidak ditemukan (`house_price_pipeline.joblib` atau `house_price_pipeline.pkl`). Silakan jalankan `python train_and_export.py` terlebih dahulu untuk melatih dan mengekspor model.")
    st.stop()

# Ekstraksi komponen bundle
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
    st.header("⚙️ Konfigurasi Cepat")
    
    st.subheader("📋 Preset Skenario Rumah")
    preset_choice = st.selectbox(
        "Pilih Contoh Profil Rumah:",
        [
            "Custom (Atur Manual)",
            "🏰 Rumah Mewah (Luxury Estate)",
            "🏡 Rumah Keluarga Nyaman (Mid-Range)",
            "🏠 Rumah Minimalis / Starter Home"
        ],
        index=0
    )
    
    # Pengaturan nilai berdasarkan preset
    if preset_choice == "🏰 Rumah Mewah (Luxury Estate)":
        p_qual, p_cond, p_year, p_remod = 9, 7, 2010, 2015
        p_grliv, p_bsmt, p_garage_cars, p_garage_area = 2450, 1450, 3, 850
        p_bath, p_bed = 3, 4
        p_neigh, p_exter, p_kitchen, p_air = "NridgHt", "Gd", "Gd", "Y"
    elif preset_choice == "🏡 Rumah Keluarga Nyaman (Mid-Range)":
        p_qual, p_cond, p_year, p_remod = 7, 6, 2002, 2005
        p_grliv, p_bsmt, p_garage_cars, p_garage_area = 1750, 950, 2, 480
        p_bath, p_bed = 2, 3
        p_neigh, p_exter, p_kitchen, p_air = "CollgCr", "TA", "TA", "Y"
    elif preset_choice == "🏠 Rumah Minimalis / Starter Home":
        p_qual, p_cond, p_year, p_remod = 5, 5, 1978, 1985
        p_grliv, p_bsmt, p_garage_cars, p_garage_area = 980, 650, 1, 280
        p_bath, p_bed = 1, 2
        p_neigh, p_exter, p_kitchen, p_air = "Edwards", "TA", "TA", "Y"
    else:
        # Default custom
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
    st.subheader("💱 Konversi Mata Uang")
    kurs_usd_idr = st.number_input(
        "Kurs USD ke IDR (Rp):",
        min_value=10000,
        max_value=25000,
        value=16000,
        step=500
    )

    st.markdown("---")
    st.caption(f"📁 **Artifact loaded:** `{bundle.get('_loaded_from', 'house_price_pipeline.joblib')}`")
    st.caption("🏷️ Model: Gradient Boosting Regressor (150 Trees, Depth 3)")

# ------------------------------------------------------------------
# 5. TAB NAVIGASI UTAMA
# ------------------------------------------------------------------
tab_prediksi, tab_analisis, tab_info = st.tabs([
    "🔮 Prediksi Harga Rumah",
    "📊 Analisis & Feature Importance",
    "ℹ️ Info Model & Penggunaan"
])

# ==================================================================
# TAB 1: FORMULIR PREDIKSI INTERAKTIF
# ==================================================================
with tab_prediksi:
    st.subheader("📝 Masukkan Karakteristik & Spesifikasi Properti")
    st.write("Sesuaikan parameter utama yang paling memengaruhi harga jual properti:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏗️ Kualitas Bangunan & Waktu")
        input_qual = st.slider("⭐ Kualitas Material Keseluruhan (OverallQual):", 1, 10, p_qual, help="1 = Buruk Sekali, 10 = Sangat Mewah / Sempurna")
        input_cond = st.slider("🛠️ Kondisi Fisik Rumah (OverallCond):", 1, 10, p_cond, help="Kondisi perawatan rumah saat ini")
        
        c_yr1, c_yr2 = st.columns(2)
        with c_yr1:
            input_year = st.number_input("📅 Tahun Dibangun (YearBuilt):", min_value=1870, max_value=2015, value=p_year)
        with c_yr2:
            input_remod = st.number_input("🔨 Tahun Renovasi (YearRemodAdd):", min_value=1950, max_value=2015, value=p_remod)

        st.markdown("#### 📍 Lokasi & Kualitas Komponen")
        neigh_options = cat_options.get("Neighborhood", ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt"])
        input_neigh = st.selectbox(
            "🏘️ Kawasan / Lingkungan (Neighborhood):",
            neigh_options,
            index=neigh_options.index(p_neigh) if p_neigh in neigh_options else 0
        )

        c_eq, c_kq = st.columns(2)
        with c_eq:
            exter_opts = cat_options.get("ExterQual", ["TA", "Gd", "Ex", "Fa"])
            input_exter = st.selectbox("🧱 Kualitas Eksterior (ExterQual):", exter_opts, index=exter_opts.index(p_exter) if p_exter in exter_opts else 0)
        with c_kq:
            kitchen_opts = cat_options.get("KitchenQual", ["TA", "Gd", "Ex", "Fa"])
            input_kitchen = st.selectbox("🍳 Kualitas Dapur (KitchenQual):", kitchen_opts, index=kitchen_opts.index(p_kitchen) if p_kitchen in kitchen_opts else 0)

        air_opts = cat_options.get("CentralAir", ["Y", "N"])
        input_air = st.radio("❄️ AC Sentral (CentralAir):", air_opts, index=air_opts.index(p_air) if p_air in air_opts else 0, horizontal=True)

    with col2:
        st.markdown("#### 📐 Dimensi & Luas Ruangan")
        input_grliv = st.slider("🏠 Luas Tempat Tinggal di Atas Tanah (GrLivArea sqft):", 300, 3000, p_grliv, help="Luas lantai hunian di atas permukaan tanah (square feet)")
        input_bsmt = st.slider("🕳️ Luas Ruang Bawah Tanah (TotalBsmtSF sqft):", 0, 2500, p_bsmt, help="Luas total basement (square feet)")

        c_gar1, c_gar2 = st.columns(2)
        with c_gar1:
            input_garage_cars = st.slider("🚗 Kapasitas Garasi Mobil (GarageCars):", 0, 4, p_garage_cars)
        with c_gar2:
            input_garage_area = st.number_input("🚘 Luas Garasi (GarageArea sqft):", min_value=0, max_value=1400, value=p_garage_area)

        st.markdown("#### 🛏️ Kamar & Sanitasi")
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            input_bath = st.slider("🚿 Kamar Mandi Lengkap (FullBath):", 0, 4, p_bath)
        with c_b2:
            input_bed = st.slider("🛏️ Kamar Tidur di Atas Tanah (BedroomAbvGr):", 0, 6, p_bed)

        input_fireplaces = st.slider("🔥 Jumlah Perapian (Fireplaces):", 0, 3, int(defaults_num.get("Fireplaces", 1)))

    # Accordion Fitur Tambahan Opsional
    with st.expander("🛠️ Pengaturan Fitur Tambahan & Detail Lainnya (Opsional)"):
        st.caption("Nilai berikut secara otomatis diisi oleh median dan modus dataset hasil cleaning.")
        c_sub1, c_sub2, c_sub3 = st.columns(3)
        with c_sub1:
            input_lotarea = st.number_input("Luas Tanah (LotArea sqft):", min_value=1000, max_value=50000, value=int(defaults_num.get("LotArea", 9500)))
            bldg_opts = cat_options.get("BldgType", ["1Fam", "TwnhsE", "Duplex", "Twnhs", "2fmCon"])
            input_bldg = st.selectbox("Tipe Bangunan (BldgType):", bldg_opts)
        with c_sub2:
            style_opts = cat_options.get("HouseStyle", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"])
            input_style = st.selectbox("Gaya Rumah (HouseStyle):", style_opts)
            input_totrms = st.slider("Total Ruangan di Atas Tanah (TotRmsAbvGrd):", 2, 14, int(defaults_num.get("TotRmsAbvGrd", 6)))
        with c_sub3:
            found_opts = cat_options.get("Foundation", ["PConc", "CBlock", "BrkTil", "Slab"])
            input_foundation = st.selectbox("Jenis Pondasi (Foundation):", found_opts)
            heat_opts = cat_options.get("HeatingQC", ["Ex", "TA", "Gd", "Fa"])
            input_heatingqc = st.selectbox("Kualitas Pemanas (HeatingQC):", heat_opts)

    st.markdown("---")
    
    # --------------------------------------------------------------
    # EKSEKUSI PREDIKSI
    # --------------------------------------------------------------
    btn_predict = st.button("🔮 Hitung Estimasi Harga Rumah", type="primary", use_container_width=True)

    # Inisialisasi dictionary fitur lengkap
    current_features = {}
    
    # 1. Isi semua dengan nilai default
    for col in feature_names:
        if col in defaults_num:
            current_features[col] = float(defaults_num[col])
        elif col in defaults_cat:
            current_features[col] = str(defaults_cat[col])
        else:
            current_features[col] = 0.0

    # 2. Timpa dengan nilai input dari form
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

    # Transformasi Fitur Sesuai StandardScaler & LabelEncoder
    processed_row = {}
    for col in feature_names:
        if col in defaults_num and col in scaler_idx:
            val = float(current_features[col])
            idx = scaler_idx[col]
            # Z-Score Standardisasi: (x - mean) / scale
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

    # Prediksi Model
    df_input = pd.DataFrame([processed_row])[feature_names]
    pred_standardized = float(model.predict(df_input)[0])
    # Rekonstruksi ke Dollar Asli ($ USD)
    predicted_usd = (pred_standardized * sp_scale) + sp_mean
    predicted_idr = predicted_usd * kurs_usd_idr

    # Margin Error MAE ($11,870)
    mae_margin = float(metrics.get("mae_usd", 11870.68))
    low_usd = max(0, predicted_usd - mae_margin)
    high_usd = predicted_usd + mae_margin
    low_idr = low_usd * kurs_usd_idr
    high_idr = high_usd * kurs_usd_idr

    # Kategori Properti
    if predicted_usd >= 280000:
        tier_label = "🏰 Properti Mewah (Luxury Tier)"
        tier_color = "#38bdf8"
    elif predicted_usd >= 180000:
        tier_label = "🏡 Properti Menengah Atas (Upper Mid-Tier)"
        tier_color = "#34d399"
    elif predicted_usd >= 130000:
        tier_label = "🏠 Properti Residensial Standar (Standard Family Home)"
        tier_color = "#fbbf24"
    else:
        tier_label = "🏷️ Properti Ekonomis / Starter (Budget Tier)"
        tier_color = "#f87171"

    # Tampilkan Hasil Prediksi
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size: 1rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; font-weight: 600;">
            ESTIMASI NILAI JUAL RUMAH (PREDICTED SALE PRICE)
        </div>
        <div class="price-usd">${predicted_usd:,.2f}</div>
        <div class="price-idr">≈ Rp {predicted_idr:,.0f} IDR</div>
        <div style="margin-top: 12px; font-size: 0.95rem; color: #cbd5e1;">
            <b>Rentang Keyakinan Model (± MAE ${mae_margin:,.2f}):</b><br>
            <span style="color: #38bdf8;">${low_usd:,.2f}</span> &nbsp;—&nbsp; <span style="color: #38bdf8;">${high_usd:,.2f}</span><br>
            <small style="color: #94a3b8;">(Rp {low_idr:,.0f} — Rp {high_idr:,.0f})</small>
        </div>
        <div style="margin-top: 16px;">
            <span style="background: rgba(255,255,255,0.1); border: 1px solid {tier_color}; color: {tier_color}; padding: 6px 16px; border-radius: 999px; font-weight: 600; font-size: 0.85rem;">
                {tier_label}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 2: ANALISIS KEPENTINGAN FITUR
# ==================================================================
with tab_analisis:
    st.subheader("📊 Analisis Kepentingan Fitur (Feature Importance)")
    st.write("Visualisasi faktor-faktor fisik dan lokasi properti yang paling dominan memengaruhi harga pasar:")

    df_imp = pd.DataFrame(top_features_list)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color="#38bdf8", height=0.65)
    ax.bar_label(bars, fmt="%.3f", padding=5, color="white", fontsize=9, fontweight="bold")

    ax.set_title("Top 15 Fitur Paling Berpengaruh Terhadap SalePrice", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Tingkat Kontribusi Relatif (Feature Importance)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#cbd5e1")
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.3, color="#64748b")
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown("""
    ### 💡 Ringkasan Temuan Ekonomi & Properti:
    1. **Kualitas Material (`OverallQual`):** Menjadi penentu tunggal nomor satu. Rumah dengan material konstruksi grade 8-10 memiliki lonjakan nilai pasar eksponensial.
    2. **Luas Hunian (`GrLivArea`):** Setiap penambahan luas ruang hunian di atas tanah secara linier meningkatkan valuasi rumah.
    3. **Ruang Bawah Tanah (`TotalBsmtSF`):** Basement yang luas di wilayah Iowa berfungsi sebagai ruang proteksi cuaca ekstrem sekaligus area rekreasi keluarga.
    4. **Fasilitas Garasi (`GarageCars` & `GarageArea`):** Kapasitas garasi untuk 2–3 mobil merupakan syarat standar bagi pembeli kelas menengah ke atas.
    5. **Usia Bangunan (`YearBuilt`):** Bangunan modern yang dibangun setelah tahun 2000 memiliki nilai depresiasi jauh lebih rendah dan efisiensi termal lebih baik.
    """)

# ==================================================================
# TAB 3: INFORMASI MODEL & PENGGUNAAN JOBLIB / PICKLE
# ==================================================================
with tab_info:
    st.subheader("ℹ️ Spesifikasi Model Machine Learning")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test R² Score", f"{metrics.get('r2', 0.8915):.4f}", "+0.8915 vs Baseline")
    with col_m2:
        st.metric("Test RMSE", f"${metrics.get('rmse_usd', 16299.45):,.2f}")
    with col_m3:
        st.metric("Test MAE", f"${metrics.get('mae_usd', 11870.68):,.2f}")
    with col_m4:
        st.metric("Test MAPE", f"{metrics.get('mape', 7.73):.2f}%")

    st.markdown("---")
    st.subheader("📦 Cara Menggunakan Artifact Joblib dan Pickle")
    st.write("Model telah diekspor ke format serialisasi standar Python. Anda dapat memuatnya dalam skrip Python atau API eksternal:")

    code_snippet = """# 1. Menggunakan JOBLIB
import joblib

# Muat model regressor langsung
model = joblib.load('model.joblib')
# Atau muat bundle lengkap (termasuk scaler & encoder)
pipeline = joblib.load('house_price_pipeline.joblib')

# 2. Menggunakan PICKLE
import pickle

with open('model.pkl', 'rb') as f:
    model_pickle = pickle.load(f)

with open('house_price_pipeline.pkl', 'rb') as f:
    pipeline_pickle = pickle.load(f)

# Prediksi menggunakan pipeline
scaled_features = ... # 75 fitur yang telah distandardisasi
y_norm = pipeline['model'].predict(scaled_features)
sale_price_usd = y_norm * pipeline['sp_scale'] + pipeline['sp_mean']
print(f"Hasil Prediksi: ${sale_price_usd[0]:,.2f}")
"""
    st.code(code_snippet, language="python")

    st.markdown("---")
    st.subheader("📑 Alur Pipeline Pra-Pemrosesan")
    st.markdown("""
    1. **Imputasi Missing Values:** Fitur numerik diimputasi dengan median, fitur kategorikal dengan modus.
    2. **Eliminasi Fitur Kosong (>75%):** Kolom `Alley`, `PoolQC`, `Fence`, dan `MiscFeature` dihapus.
    3. **Pembersihan Outlier (IQR):** Menghapus data ekstrem dengan batas $1.5 \\times \\text{IQR}$, menyaring dataset menjadi 557 baris bersih.
    4. **Standardisasi Fitur Numerik:** Menggunakan `StandardScaler` (Z-Score: mean $\\approx 0$, std $\\approx 1$).
    5. **Encoding Fitur Kategorikal:** Menggunakan `LabelEncoder` pada 39 kolom kategorik.
    6. **Split Latih/Uji:** 80% Latih (445 sampel) dan 20% Uji (112 sampel).
    """)

# ------------------------------------------------------------------
# 6. DUAL RUNNER (STREAMLIT / PYTHON CLI)
# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
