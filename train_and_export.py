"""
Script Pelatihan dan Ekspor Model Prediksi Harga Rumah (Ames Housing)
Menghasilkan artifact model dalam format .joblib dan .pkl
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_and_export():
    print("=" * 60)
    print("1. MEMUAT DATASET & AUDIT AWAL")
    print("=" * 60)
    data_path = "train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} tidak ditemukan!")
        
    train = pd.read_csv(data_path)
    print(f"Dataset dimuat: {train.shape[0]} baris, {train.shape[1]} kolom")

    print("\n" + "=" * 60)
    print("2. PREPROCESSING & DATA CLEANING")
    print("=" * 60)
    missing_values = train.isnull().sum()
    less = missing_values[missing_values < 1000].index
    over = missing_values[missing_values >= 1000].index

    # Imputasi Missing Value
    numeric_features = train[less].select_dtypes(include=[np.number]).columns
    train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

    kategorical_features = train[less].select_dtypes(include=["object"]).columns
    for column in kategorical_features:
        train[column] = train[column].fillna(train[column].mode()[0])

    # Hapus kolom dengan missing >= 1000
    df = train.drop(columns=over)
    print(f"Kolom dibuang (missing >= 1000): {list(over)}")

    # Simpan nilai default (median & mode) sebelum transformasi skala untuk keperluan input frontend
    defaults_numeric = {col: float(df[col].median()) for col in numeric_features if col not in ['SalePrice', 'Id']}
    defaults_categorical = {col: str(df[col].mode()[0]) for col in df.select_dtypes(include=["object"]).columns}

    # Penanganan Outlier dengan IQR
    Q1 = df[numeric_features].quantile(0.25)
    Q3 = df[numeric_features].quantile(0.75)
    IQR = Q3 - Q1
    condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)

    df_filtered_numeric = df.loc[condition, numeric_features]
    categorical_features = df.select_dtypes(include=["object"]).columns
    df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)
    print(f"Baris setelah pembersihan outlier: {len(df)} baris")

    # Standardisasi StandardScaler
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    df = df.drop_duplicates()

    # Konstanta target SalePrice
    sp_idx = list(numeric_features).index('SalePrice')
    sp_mean = float(scaler.mean_[sp_idx])
    sp_scale = float(scaler.scale_[sp_idx])
    print(f"Konstanta Rekonstruksi Target SalePrice: Mean=${sp_mean:,.2f}, Scale=${sp_scale:,.2f}")

    # Label Encoding
    label_encoders = {}
    categorical_options = {}
    df_lencoder = df.copy()
    for col in categorical_features:
        le = LabelEncoder()
        df_lencoder[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        categorical_options[col] = [str(c) for c in le.classes_]

    # Train Test Split
    X = df_lencoder.drop(columns=["SalePrice"])
    y = df_lencoder["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=1
    )

    X_train_model = X_train.drop(columns=['Id']) if 'Id' in X_train.columns else X_train.copy()
    X_test_model = X_test.drop(columns=['Id']) if 'Id' in X_test.columns else X_test.copy()
    feature_names = list(X_train_model.columns)
    print(f"Jumlah fitur prediktor: {len(feature_names)}")

    print("\n" + "=" * 60)
    print("3. PELATIHAN MODEL (GRADIENT BOOSTING TUNED)")
    print("=" * 60)
    best_model = GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=150,
        subsample=0.8,
        random_state=42
    )
    best_model.fit(X_train_model, y_train)

    # Evaluasi
    y_pred_test = best_model.predict(X_test_model)
    y_test_usd = y_test * sp_scale + sp_mean
    y_pred_usd = y_pred_test * sp_scale + sp_mean

    r2 = r2_score(y_test, y_pred_test)
    rmse_usd = np.sqrt(mean_squared_error(y_test_usd, y_pred_usd))
    mae_usd = mean_absolute_error(y_test_usd, y_pred_usd)
    mape = np.mean(np.abs((y_test_usd - y_pred_usd) / y_test_usd)) * 100

    print(f"Test R^2 Score : {r2:.4f}")
    print(f"Test RMSE ($)  : ${rmse_usd:,.2f}")
    print(f"Test MAE ($)   : ${mae_usd:,.2f}")
    print(f"Test MAPE (%)  : {mape:.2f}%")

    # Feature Importance
    importances = best_model.feature_importances_
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    top_features = df_importance.head(15).to_dict(orient="records")

    print("\n" + "=" * 60)
    print("4. EKSPOR KE JOBLIB DAN PICKLE")
    print("=" * 60)
    
    # 1. Standalone Model
    joblib_model_path = "model.joblib"
    pickle_model_path = "model.pkl"
    joblib.dump(best_model, joblib_model_path)
    with open(pickle_model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"[OK] Standalone model disimpan ke: {joblib_model_path} ({os.path.getsize(joblib_model_path):,} bytes)")
    print(f"[OK] Standalone model disimpan ke: {pickle_model_path} ({os.path.getsize(pickle_model_path):,} bytes)")

    # 2. Pipeline Bundle (Model + Metadata Preprocessing untuk Frontend)
    # Peta indeks scaler untuk fitur numerik
    numeric_cols_model = [col for col in feature_names if col in numeric_features]
    scaler_feature_indices = {col: list(numeric_features).index(col) for col in numeric_cols_model}

    bundle = {
        "model": best_model,
        "model_name": "Gradient Boosting Regressor (Tuned)",
        "feature_names": feature_names,
        "numeric_features": numeric_cols_model,
        "categorical_features": list(categorical_features),
        "categorical_options": categorical_options,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "scaler_feature_indices": scaler_feature_indices,
        "sp_mean": sp_mean,
        "sp_scale": sp_scale,
        "defaults_numeric": defaults_numeric,
        "defaults_categorical": defaults_categorical,
        "top_features": top_features,
        "metrics": {
            "r2": r2,
            "rmse_usd": rmse_usd,
            "mae_usd": mae_usd,
            "mape": mape
        }
    }

    joblib_bundle_path = "house_price_pipeline.joblib"
    pickle_bundle_path = "house_price_pipeline.pkl"
    joblib.dump(bundle, joblib_bundle_path)
    with open(pickle_bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[OK] Pipeline bundle disimpan ke: {joblib_bundle_path} ({os.path.getsize(joblib_bundle_path):,} bytes)")
    print(f"[OK] Pipeline bundle disimpan ke: {pickle_bundle_path} ({os.path.getsize(pickle_bundle_path):,} bytes)")

    print("\n" + "=" * 60)
    print("EKSPOR SELESAI DENGAN SUKSES!")
    print("=" * 60)

if __name__ == "__main__":
    train_and_export()
