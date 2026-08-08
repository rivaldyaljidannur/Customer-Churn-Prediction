# 📊 Customer Churn Prediction

Project Machine Learning untuk memprediksi kemungkinan pelanggan melakukan **churn** atau berhenti menggunakan layanan perusahaan telekomunikasi berdasarkan karakteristik pelanggan.

Model yang digunakan dalam project ini adalah **Random Forest Classifier**, dengan proses mulai dari eksplorasi data, preprocessing, pembangunan model, evaluasi, hingga deployment menggunakan Streamlit.

## 🌐 Live Application

Aplikasi Customer Churn Prediction dapat diakses melalui:

https://customer-churn-prediction-tel.streamlit.app/

---

## 📌 Project Overview

Customer churn merupakan kondisi ketika pelanggan berhenti menggunakan layanan suatu perusahaan. Tingginya tingkat churn dapat memberikan dampak terhadap pendapatan dan pertumbuhan bisnis.

Project ini bertujuan untuk membangun model Machine Learning yang dapat membantu mengidentifikasi pelanggan yang memiliki kemungkinan untuk melakukan churn berdasarkan informasi seperti:

- Lama berlangganan (`tenure`)
- Jenis kontrak
- Layanan internet
- Metode pembayaran
- Monthly Charges
- Total Charges
- Tech Support
- Online Security
- dan karakteristik pelanggan lainnya.

Hasil prediksi dibagi menjadi dua kelas:

| Label | Keterangan |
|---|---|
| 0 | No Churn |
| 1 | Churn |

---

## 🎯 Objectives

Tujuan utama project ini adalah:

- Melakukan eksplorasi terhadap karakteristik pelanggan.
- Mengidentifikasi pola yang berkaitan dengan customer churn.
- Melakukan preprocessing terhadap data kategorikal dan numerik.
- Membangun model klasifikasi menggunakan Random Forest.
- Mengevaluasi kemampuan model dalam mendeteksi pelanggan churn.
- Mengimplementasikan model ke dalam aplikasi berbasis Streamlit.

---

## 📂 Dataset

Dataset yang digunakan adalah **Telco Customer Churn Dataset** yang berisi informasi mengenai pelanggan perusahaan telekomunikasi.

Beberapa variabel yang tersedia antara lain:

| Feature | Description |
|---|---|
| `customerID` | ID unik pelanggan |
| `gender` | Jenis kelamin pelanggan |
| `SeniorCitizen` | Status pelanggan senior |
| `Partner` | Status memiliki pasangan |
| `Dependents` | Status memiliki tanggungan |
| `tenure` | Lama pelanggan berlangganan |
| `PhoneService` | Layanan telepon |
| `InternetService` | Jenis layanan internet |
| `OnlineSecurity` | Layanan keamanan online |
| `TechSupport` | Layanan technical support |
| `Contract` | Jenis kontrak pelanggan |
| `PaymentMethod` | Metode pembayaran |
| `MonthlyCharges` | Biaya bulanan |
| `TotalCharges` | Total biaya pelanggan |
| `Churn` | Status pelanggan churn atau tidak |

Dataset:

https://raw.githubusercontent.com/Aljidannur/Customer-Churn-Prediction/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv

---

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis (EDA) dilakukan untuk memahami karakteristik dan pola pada dataset sebelum proses modeling.

Analisis yang dilakukan meliputi:

- Distribusi target Churn
- Distribusi fitur numerik
- Analisis `tenure`
- Analisis `MonthlyCharges`
- Analisis `TotalCharges`
- Analisis fitur kategorikal terhadap Churn
- Analisis korelasi
- Pemeriksaan missing value
- Pemeriksaan data duplikat

EDA membantu memahami karakteristik pelanggan serta menentukan proses preprocessing yang diperlukan sebelum data digunakan oleh model.

---

## ⚙️ Data Preparation

Sebelum digunakan untuk Machine Learning, dataset melalui beberapa tahap preprocessing:

### 1. Data Cleaning

- Mengubah `TotalCharges` menjadi tipe data numerik.
- Menangani missing value.
- Memeriksa data duplikat.
- Menghapus fitur yang tidak diperlukan untuk modeling.

### 2. Target Encoding

Target `Churn` diubah menjadi:

```text
No  → 0
Yes → 1
```

### 3. One-Hot Encoding

Variabel kategorikal diubah menjadi bentuk numerik menggunakan One-Hot Encoding.

```python
df_prep = pd.get_dummies(df_prep, drop_first=True)
```

### 4. Feature & Target Separation

```python
X = df_prep.drop('Churn', axis=1)
y = df_prep['Churn']
```

---

## ✂️ Train-Test Split

Dataset dibagi menjadi:

- **80% Training Data**
- **20% Testing Data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Parameter `stratify=y` digunakan untuk menjaga proporsi kelas Churn dan No Churn tetap konsisten pada data training dan testing.

---

## 🌲 Machine Learning Model

Model yang digunakan adalah:

### Random Forest Classifier

Random Forest merupakan algoritma ensemble learning yang membangun banyak Decision Tree dan menggabungkan hasil prediksi setiap tree melalui proses voting.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)
```

Model mempelajari pola dari berbagai karakteristik pelanggan untuk menentukan apakah pelanggan memiliki kecenderungan untuk churn.

---

## 📊 Model Evaluation

Model dievaluasi menggunakan beberapa classification metrics.

| Metric | Score |
|---|---:|
| Accuracy | **78.15%** |
| Precision | **62.26%** |
| Recall | **44.35%** |
| F1-Score | **51.81%** |
| ROC-AUC | **81.76%** |

### Interpretation

**Accuracy — 78.15%**

Model mampu memberikan prediksi yang benar terhadap sekitar 78% data testing.

**Precision — 62.26%**

Dari seluruh pelanggan yang diprediksi akan churn, sekitar 62% benar-benar merupakan pelanggan churn.

**Recall — 44.35%**

Model berhasil mendeteksi sekitar 44% dari seluruh pelanggan yang benar-benar churn. Nilai ini menunjukkan bahwa masih terdapat pelanggan churn yang belum berhasil terdeteksi oleh model.

**F1-Score — 51.81%**

Menunjukkan keseimbangan antara Precision dan Recall model.

**ROC-AUC — 81.76%**

Model memiliki kemampuan yang baik dalam membedakan pelanggan yang berpotensi churn dan tidak churn.

---

## 🔍 Feature Importance

Random Forest menyediakan Feature Importance yang dapat digunakan untuk mengetahui fitur mana yang paling banyak berkontribusi terhadap proses pengambilan keputusan model.

```python
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(
    by='Importance',
    ascending=False
)
```

Analisis ini membantu memahami karakteristik pelanggan yang memiliki kontribusi lebih besar terhadap prediksi churn.

---

## 🔄 Machine Learning Workflow

```text
Business Understanding
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Data Preparation
        ↓
Feature & Target Selection
        ↓
Train-Test Split
        ↓
Random Forest Classifier
        ↓
Prediction
        ↓
Model Evaluation
        ↓
Feature Importance
        ↓
Deployment
```

---

## 🚀 Deployment

Model diimplementasikan ke dalam aplikasi web menggunakan **Streamlit**.

Aplikasi memungkinkan pengguna memasukkan karakteristik pelanggan dan mendapatkan hasil prediksi apakah pelanggan tersebut memiliki kecenderungan:

```text
Customer Churn
```

atau

```text
Customer Not Churn
```

### 🔗 Live Demo

https://customer-churn-prediction-tel.streamlit.app/

---

## 🛠️ Tech Stack

Project ini dikembangkan menggunakan:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Random Forest
- Streamlit
- Joblib
- Jupyter Notebook / Google Colab
- GitHub

---

## 📦 Installation

Clone repository:

```bash
git clone https://github.com/Aljidannur/Customer-Churn-Prediction.git
```

Masuk ke directory project:

```bash
cd Customer-Churn-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

---

## 📋 Requirements

Contoh library yang digunakan pada project:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
openpyxl
```

---

## 💡 Business Value

Model Customer Churn Prediction dapat membantu perusahaan mengidentifikasi pelanggan yang memiliki risiko berhenti menggunakan layanan.

Informasi tersebut dapat digunakan sebagai dasar untuk melakukan strategi retensi pelanggan seperti:

- Memberikan penawaran khusus.
- Memberikan diskon atau loyalty program.
- Meningkatkan customer support.
- Melakukan pendekatan terhadap pelanggan berisiko tinggi.
- Mengevaluasi layanan yang berkaitan dengan tingginya churn.

Dengan demikian, perusahaan dapat melakukan tindakan preventif sebelum pelanggan benar-benar berhenti menggunakan layanan.

---

## 📌 Conclusion

Berdasarkan hasil modeling menggunakan **Random Forest Classifier**, diperoleh Accuracy sebesar **78.15%** dan ROC-AUC sebesar **81.76%**.

Nilai ROC-AUC menunjukkan bahwa model memiliki kemampuan yang baik dalam membedakan pelanggan churn dan tidak churn. Namun, nilai Recall sebesar **44.35%** menunjukkan bahwa kemampuan model dalam mendeteksi seluruh pelanggan yang benar-benar churn masih dapat ditingkatkan.

Pengembangan selanjutnya dapat dilakukan melalui:

- Hyperparameter tuning
- Class weighting
- Threshold optimization
- Resampling pada data tidak seimbang
- Perbandingan dengan algoritma klasifikasi lainnya

---

## 👨‍💻 Author

**Andi Muhammad Rivaldy Aljidannur**

Information Systems Graduate | Data Analytics | Machine Learning | Business Intelligence
