# Proyek Machine Learning: Prediksi Harga Rumah (House Prices)
> **Pipeline End-to-End: Data Cleaning, Preprocessing, Exploratory Data Analysis (EDA), dan Pemodelan Regresi**

---

## Ringkasan Proyek
Proyek ini merupakan dokumentasi komprehensif dari proses belajar kami dalam membangun pipeline *machine learning* regresi dari tahap data mentah (*raw data*) hingga siap produksi (*production-ready*). Studi kasus yang diangkat adalah memprediksi harga jual properti/rumah (`SalePrice`) berdasarkan karakteristik fisik, fasilitas, dan lokasi bangunan menggunakan **Ames Housing Dataset**.

Dataset mentah memiliki berbagai tantangan nyata seperti *missing value* tinggi, *outlier* ekstrem, perbedaan skala fitur numerik, serta fitur kategorikal yang kompleks. Kami mendemonstrasikan bagaimana teknik preprocessing yang tepat secara langsung meningkatkan performa generalisasi model machine learning.

---

## Struktur File dan Folder

```text
├── pertemuan6_data_cleaning_transformation.ipynb   # Notebook utama (Data Cleaning, EDA, Modeling, Export)
├── app.py                                          # Frontend aplikasi web interaktif (Streamlit)
├── train_and_export.py                             # Skrip otomatisasi pelatihan & ekspor model
├── model.joblib                                    # Model Gradient Boosting Regressor (Joblib)
├── model.pkl                                       # Model Gradient Boosting Regressor (Pickle)
├── house_price_pipeline.joblib                     # Pipeline lengkap (Model + Scaler + Encoder + Meta)
├── house_price_pipeline.pkl                        # Pipeline lengkap format Pickle
├── requirements.txt                                # Daftar dependensi Python untuk deployment
├── train.csv                                       # Dataset latihan mentah (1.460 baris, 81 kolom)
├── test.csv                                        # Dataset pengujian (1.459 baris, 80 kolom)
├── sample_submission.csv                           # Format penyerahan prediksi kompetisi
└── README.md                                       # Dokumentasi lengkap proyek
```

---

## Tahapan Alur Kerja (Pipeline Workflow)

Pipeline pengolahan data dan pemodelan pada proyek ini dibagi menjadi 15 tahapan terstruktur:

### 1. Eksplorasi Awal & Audit Data
- **Dataset:** 1.460 baris dan 81 kolom (38 fitur numerik, 43 fitur kategorikal).
- **Audit Tipe Data:** Memastikan kesesuaian tipe data dan mengidentifikasi kolom integer yang otomatis terbaca sebagai `float64` akibat adanya nilai `NaN`.

### 2. Penanganan Missing Value
- **Missing > 75%:** Dihapus (*drop*) sebanyak 4 fitur (`Alley`, `PoolQC`, `Fence`, `MiscFeature`) karena tingkat kekosongannya melebihi 1.000 baris dan berisiko menimbulkan bias imputasi.
- **Missing < 75% (Numerik):** Diimputasi menggunakan **Median** per kolom karena lebih tahan terhadap nilai ekstrem dibandingkan Mean.
- **Missing < 75% (Kategorikal):** Diimputasi menggunakan **Modus** (nilai/kategori frekuensi tertinggi).

### 3. Deteksi & Pembersihan Outlier (Metode IQR)
- Menggunakan visualisasi **Boxplot** untuk mendeteksi sebaran titik di luar rentang kuartil.
- Menghitung **Interquartile Range**:
  $$\text{IQR} = Q3 - Q1$$
  - Batas Bawah: $Q1 - 1.5 \times \text{IQR}$
  - Batas Atas: $Q3 + 1.5 \times \text{IQR}$
- Baris data yang berada di luar rentang batas pada fitur numerik dieliminasi, menghasilkan dataset yang lebih homogen dan bersih (557 baris).

### 4. Standardisasi Skala Fitur (Z-Score Scaling)
- Rentang nilai antar fitur numerik sangat bervariasi (mulai dari biner 0–1 hingga puluhan ribu pada `LotArea`).
- Menggunakan `StandardScaler` dari Scikit-Learn:
  $$z = \frac{x - \mu}{\sigma}$$
  sehingga seluruh fitur numerik terpusat pada mean $\approx 0$ dan standar deviasi $\approx 1$.

### 5. Pemeriksaan Data Duplikat
- Menggunakan `df.duplicated().sum()`. Terverifikasi bahwa tidak ada baris duplikat yang identik (0 duplikat).

### 6. Rekayasa Fitur Kategorikal (Feature Encoding)
- Menguji dua pendekatan:
  1. **One-Hot Encoding (`pd.get_dummies`):** Menghasilkan lonjakan dimensi menjadi ratusan kolom (*curse of dimensionality*).
  2. **Label Encoding (`LabelEncoder`):** Mengonversi kategori menjadi integer urut tanpa menambah jumlah kolom (tetap 77 kolom).
- Dataframe hasil Label Encoding (`df_lencoder`) dipilih untuk tahapan analisis dan pemodelan lanjutan.

### 7. Exploratory Data Analysis (EDA) & Analisis Korelasi
- Memverifikasi ulang bahwa data 100% bebas dari missing value.
- Visualisasi sebaran histogram seluruh kolom.
- Menghitung **Korelasi Pearson** terhadap variabel target `SalePrice`. Fitur penentu utama dengan korelasi linier terkuat:
  1. `OverallQual` ($r = 0.79$) — Kualitas material dan pengerjaan rumah.
  2. `GrLivArea` ($r = 0.71$) — Luas area tempat tinggal di atas tanah.
  3. `GarageCars` ($r = 0.64$) & `GarageArea` ($r = 0.62$) — Fasilitas garasi kendaraan.
  4. `TotalBsmtSF` ($r = 0.61$) — Luas ruang bawah tanah (basement).
  5. `YearBuilt` ($r = 0.52$) — Tahun pembuatan konstruksi bangunan.

### 8. Pembagian Data (Train-Test Split)
- Data dibagi dengan rasio **80% Data Latih (445 sampel)** dan **20% Data Uji (112 sampel)** menggunakan `random_state=1`.
- Verifikasi kurva Kernel Density Estimation (KDE) mengonfirmasi bahwa distribusi target pada data latih dan data uji sangat identik (bebas dari sampling bias).

### 9. Pemodelan Machine Learning
Menguji 7 algoritma dengan paradigma berbeda:
1. **DummyRegressor (Mean Baseline):** Patokan tebakan naif rata-rata harga.
2. **Linear Regression (OLS):** Regresi parametrik linier klasik.
3. **Ridge Regression (L2):** Regularisasi kuadrat bobot untuk menekan multikolinearitas.
4. **Lasso Regression (L1):** Regularisasi nilai mutlak yang menyeleksi fitur otomatis.
5. **Random Forest Regressor:** Ensemble pohon keputusan berbasis Bagging.
6. **Gradient Boosting Regressor:** Ensemble sekuensial Boosting.
7. **Support Vector Regression (SVR - RBF Kernel):** Model regresi berbasis hyperplane dan margin.

---

## Hasil Evaluasi Model

### 1. Komparasi Kinerja pada Data Uji (Skala Terstandardisasi & Satuan Asli $ USD)

| Model | CV $R^2$ (Train) | Test $R^2$ | Test RMSE (Z-Score) | Test RMSE ($ USD) | Test MAE ($ USD) | Test MAPE (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Dummy)** | -0.0125 | -0.0258 | 0.9075 | $50.108,41 | $39.906,02 | 26.68% |
| **Linear Regression** | 0.8124 | 0.8791 | 0.3117 | $17.208,45 | $12.356,80 | 8.01% |
| **Ridge Regression** | 0.8159 | 0.8846 | 0.3045 | $16.812,79 | $12.189,45 | 7.91% |
| **Lasso Regression** | 0.8142 | 0.8816 | 0.3084 | $17.027,33 | $12.062,11 | **7.78%** |
| **Random Forest** | 0.8351 | 0.8890 | 0.2987 | $16.491,12 | $12.221,05 | 7.95% |
| **Gradient Boosting** | **0.8412** | **0.8898** | **0.2974** | **$16.421,55** | **$12.055,14** | **7.79%** |
| **SVR (RBF Kernel)** | 0.8298 | 0.8654 | 0.3288 | $18.155,20 | $12.870,14 | 8.35% |

### 2. Hasil Hyperparameter Tuning (GridSearchCV pada Gradient Boosting)
Melalui pencarian grid pada hyperparameter:
- `learning_rate`: **0.05**
- `n_estimators`: **150**
- `max_depth`: **3**
- `subsample`: **0.8**

**Hasil Peningkatan Model Terbaik:**
- **Test $R^2$ Score:** Naik menjadi **0.8915** (mampu menjelaskan 89.15% varians harga rumah).
- **Test RMSE ($ USD):** Turun menjadi **$16.303,46**.
- **Test MAE ($ USD):** Turun menjadi **$12.042,87**.
- **Test MAPE:** Turun menjadi **7.76%** (kategori *high precision valuation*).

---

## Rujukan Ilmiah: Interpretasi Metrik Evaluasi ($R^2$, RMSE, MAE)

Dalam mengevaluasi performa model regresi pada proyek ini, interpretasi metrik didasarkan pada landasan literatur ilmiah berikut:

> **Prinsip Utama:**  
> Semakin mendekati **1 nilai $R^2$**, serta semakin mendekati **0 nilai RMSE dan MAE**, menunjukkan kinerja model yang semakin baik (*Chicco et al., 2021; Yasin et al., 2025; Zarbakhsh et al., 2024*).

### 1. Publikasi Internasional (Scopus Q1) - Landasan Teori Evaluasi Metrik
**Sitasi:**  
Chicco, D., Warrens, M. J., & Jurman, G. (2021). The coefficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation. *PeerJ Computer Science, 7*, e623. [https://doi.org/10.7717/peerj-cs.623](https://doi.org/10.7717/peerj-cs.623)
- **$R^2$ (Koefisien Determinasi):** Rentang nilai dari $-\infty$ (terburuk) hingga $+1$ (kecocokan sempurna). Nilai 1 berarti model mampu memprediksi target dengan presisi tanpa kesalahan.
- **RMSE & MAE:** Rentang nilai dari $0$ (sempurna tanpa error) hingga $+\infty$ (terburuk). Nilai yang makin besar menunjukkan model yang semakin meleset.
- **Catatan Penting:** Besaran mutlak MAE atau RMSE harus selalu dievaluasi terhadap skala data aslinya. Oleh karena itu, kami menyajikan metrik dalam skala terstandardisasi maupun dalam nilai mata uang riil ($ USD) dan persentase relatif (MAPE).

### 2. Publikasi Internasional (Scopus Q1) - Komparasi Kinerja Model Ensemble
**Sitasi:**  
Zarbakhsh, S., Shahsavar, A. R., & Soltani, M. (2024). Optimizing PGRs for in vitro shoot proliferation of pomegranate with bayesian-tuned ensemble stacking regression and NSGA-II: A comparative evaluation of machine learning models. *Plant Methods, 20*, Article 82. [https://doi.org/10.1186/s13007-024-01211-5](https://doi.org/10.1186/s13007-024-01211-5)
- **$R^2$ (Koefisien Determinasi):** Semakin dekat nilai $R^2$ ke 1, semakin baik kapasitas model dalam menerangkan variabilitas data.
- **RMSE & MAE:** Nilai metrik yang semakin rendah merefleksikan performa prediksi yang lebih akurat.
- **Kriteria Model Terbaik:** Model terbaik ditentukan oleh kombinasi nilai $R^2$ yang lebih tinggi serta nilai metrik error (RMSE, MAE, dan MAPE) yang lebih kecil pada data latih maupun data uji.

### 3. Publikasi Nasional (SINTA 2) - Evaluasi Pemodelan Prediksi
**Sitasi:**  
Yasin, F., Firmansyah, M. R., Aldo, D., & Amrustian, M. A. (2025). Multivariate forecasting of paddy production: A comparative study of machine learning models. *Jurnal Teknik Informatika (JUTIF), 6*(3), 1431–1442. [https://doi.org/10.52436/1.jutif.2025.6.3.4681](https://doi.org/10.52436/1.jutif.2025.6.3.4681)
- **MAE:** Semakin rendah nilai MAE, semakin baik akurasi estimasi model.
- **RMSE:** Memberikan penalti kuadrat yang lebih besar pada error ekstrem; nilai yang semakin rendah menunjukkan performa model yang semakin solid.
- **$R^2$:** Mengukur proporsi variabilitas target yang berhasil diterangkan oleh kombinasi fitur prediktor.

---


## Temuan Utama & Kesimpulan Proyek

1. **Efektivitas Data Preprocessing:**  
   Pembersihan missing value terarah, eliminasi outlier dengan IQR, dan standardisasi fitur berkontribusi langsung menaikkan $R^2$ model linier dan ensemble ke rentang $0.88 - 0.89$.
2. **Model Terbaik untuk Estimasi Nilai Rumah:**  
   Model **Gradient Boosting Regressor (Tuned)** memberikan performa tertinggi dengan rata-rata kesalahan prediksi hanya **\$12.042 (7.76%)**, memangkas error baseline naif sebesar **~70%**.
3. **Pilar Penentu Harga Properti (Feature Importance):**  
   Faktor fisik yang paling menentukan nilai jual rumah di Ames secara berurutan adalah:
   - Kualitas material bangunan & finishing (`OverallQual`)
   - Luas total area ruang huni di atas tanah (`GrLivArea`)
   - Fasilitas dan kapasitas kendaraan pada garasi (`GarageCars`, `GarageArea`)
   - Luas ruangan basement (`TotalBsmtSF`)
   - Tahun awal pembuatan rumah (`YearBuilt`)
4. **Rencana Pengembangan Selanjutnya:**
   - Eksperimen transformasi logaritmik $\log(1 + y)$ pada target `SalePrice`.
   - Feature engineering rasio ruangan gabungan (`TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF`).
   - Menerapkan arsitektur *Stacking Regressor* (menggabungkan prediksi Ridge, Random Forest, dan Gradient Boosting).

---

## Deployment & Frontend Interaktif (Streamlit Web App)

Proyek ini telah dilengkapi dengan model yang diekspor ke format serialisasi standar industri (**Joblib** dan **Pickle**) serta aplikasi web frontend interaktif berbasis **Streamlit** untuk memfasilitasi pengujian inferensi harga rumah secara langsung.

### 1. Struktur Artifact Model Serialisasi
Model dan pipeline pendukung diekspor dalam 2 varian format:
- `model.joblib` & `model.pkl`: Model estimator `GradientBoostingRegressor` murni hasil tuning.
- `house_price_pipeline.joblib` & `house_price_pipeline.pkl`: Bundle komprehensif berisi model, objek `StandardScaler`, daftar `LabelEncoder`, konstanta pemulihan skala (`sp_mean`, `sp_scale`), nilai default median/mode, dan metrik performa.

### 2. Cara Menjalankan Aplikasi Web Frontend
Pastikan dependensi telah terpasang, lalu jalankan antarmuka Streamlit:

```bash
# 1. Instalasi dependensi
pip install -r requirements.txt

# 2. Jalankan aplikasi web
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka di peramban (browser) pada alamat `http://localhost:8501`.

### 3. Fitur Utama Frontend Web:
- **Preset Skenario Instan:** Pilihan cepat profil rumah (Mewah, Keluarga Nyaman, Minimalis, atau Kustom).
- **Formulir Interaktif:** Kontrol slider dan input untuk 15+ parameter paling berpengaruh (`OverallQual`, `GrLivArea`, `GarageCars`, `YearBuilt`, `Neighborhood`, dll.).
- **Kalkulasi Nilai Ganda:** Estimasi harga jual ditampilkan dalam satuan **USD ($)** dan konversi otomatis ke **Rupiah (IDR)** berdasarkan kurs yang dapat disesuaikan.
- **Rentang Keyakinan (Error Margin):** Menampilkan batas estimasi atas dan bawah berdasarkan toleransi metrik MAE ($\pm \$11.870$).
- **Visualisasi Feature Importance:** Grafik kontribusi fitur terhadap pembentukan harga pasar.

### 4. Pelatihan Ulang & Ekspor Ulang Otomatis
Untuk melatih ulang model dan memperbarui seluruh file artifact serialisasi secara otomatis:

```bash
python train_and_export.py
```

### 5. Contoh Pemanggilan Model via Python:
```python
import joblib

# Memuat pipeline bundle
pipeline = joblib.load("house_price_pipeline.joblib")
model = pipeline["model"]
sp_scale = pipeline["sp_scale"]
sp_mean = pipeline["sp_mean"]

# Prediksi data uji
# y_pred_usd = model.predict(X_scaled) * sp_scale + sp_mean
```
