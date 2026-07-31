# Customer-Churn-Prediction
# ♻️ Klasterisasi Kinerja Pengelolaan Sampah Kalimantan Timur

<p align="center">
  <b>Analisis Kinerja Pengelolaan Sampah Kabupaten/Kota di Kalimantan Timur Menggunakan K-Means Clustering</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue">
  <img src="https://img.shields.io/badge/Machine%20Learning-K--Means-orange">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red">
  <img src="https://img.shields.io/badge/Status-Completed-success">
</p>

---

## 📌 Tentang Proyek

Proyek ini merupakan implementasi **Data Science dan Machine Learning** untuk menganalisis serta mengelompokkan kinerja pengelolaan sampah pada **10 kabupaten/kota di Provinsi Kalimantan Timur** selama periode **2019–2024**.

Algoritma **K-Means Clustering** digunakan untuk mengelompokkan daerah berdasarkan karakteristik kinerja pengelolaan sampah. Jumlah cluster optimal ditentukan menggunakan **Elbow Method**, sedangkan kualitas hasil clustering dievaluasi menggunakan **Silhouette Score**.

Hasil analisis kemudian dikembangkan menjadi aplikasi web interaktif menggunakan **Streamlit** agar informasi mengenai kinerja pengelolaan sampah dapat dieksplorasi dengan lebih mudah.

---

## 🎯 Tujuan

- Menganalisis perkembangan kinerja pengelolaan sampah di Kalimantan Timur.
- Mengidentifikasi pola pengelolaan sampah antar kabupaten/kota.
- Mengelompokkan daerah berdasarkan karakteristik kinerja pengelolaan sampah.
- Menentukan jumlah cluster optimal menggunakan **Elbow Method**.
- Mengevaluasi hasil clustering menggunakan **Silhouette Score**.
- Memvisualisasikan hasil clustering dalam bentuk grafik dan peta interaktif.
- Mengembangkan dashboard menggunakan **Streamlit**.

---

## 📊 Dataset

Dataset yang digunakan merupakan data kinerja pengelolaan sampah pada **10 kabupaten/kota di Provinsi Kalimantan Timur** selama periode **2019–2024**.

### Wilayah yang Dianalisis

- Samarinda
- Balikpapan
- Bontang
- Kutai Kartanegara
- Kutai Timur
- Kutai Barat
- Berau
- Paser
- Penajam Paser Utara
- Mahakam Ulu

### Variabel Utama

| Variabel | Keterangan |
|---|---|
| Tahun | Tahun pengamatan |
| Kabupaten/Kota | Wilayah pengamatan |
| Timbulan Sampah | Jumlah timbulan sampah |
| Pengurangan Sampah | Jumlah sampah yang berhasil dikurangi |
| % Pengurangan | Persentase pengurangan sampah |
| Penanganan Sampah | Jumlah sampah yang berhasil ditangani |
| % Penanganan | Persentase penanganan sampah |
| Sampah Terkelola | Total sampah yang berhasil dikelola |
| % Sampah Terkelola | Persentase keseluruhan sampah yang dikelola |

---

## 🔄 Alur Analisis

```text
Business Understanding
        ↓
Data Understanding
        ↓
Data Preparation
        ↓
Exploratory Data Analysis
        ↓
Data Normalization
        ↓
Elbow Method
        ↓
K-Means Clustering
        ↓
Model Evaluation
        ↓
Cluster Interpretation
        ↓
Visualization & Mapping
        ↓
Streamlit Deployment
```

---

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis (EDA) dilakukan untuk memahami karakteristik dataset sebelum proses clustering.

Beberapa analisis yang dilakukan meliputi:

- Pemeriksaan struktur dan tipe data.
- Pemeriksaan missing value.
- Pemeriksaan data duplikat.
- Statistik deskriptif.
- Analisis distribusi data.
- Analisis perkembangan pengelolaan sampah berdasarkan tahun.
- Perbandingan kinerja antar kabupaten/kota.
- Analisis korelasi antar variabel.
- Visualisasi indikator pengelolaan sampah.

---

## ⚙️ Data Preparation

Sebelum proses clustering, dilakukan beberapa tahapan persiapan data:

1. Pemeriksaan kualitas data.
2. Penanganan missing value dan data duplikat.
3. Pemilihan fitur yang relevan.
4. Menghapus atribut yang tidak digunakan dalam proses clustering.
5. Normalisasi data menggunakan **MinMaxScaler**.

Normalisasi dilakukan untuk mengubah seluruh fitur numerik ke dalam rentang yang sama sehingga perbedaan skala antar variabel tidak mendominasi proses perhitungan jarak pada K-Means.

---

## 🤖 K-Means Clustering

**K-Means** merupakan algoritma *Unsupervised Machine Learning* yang mengelompokkan data berdasarkan tingkat kemiripan karakteristik.

Jumlah cluster optimal ditentukan menggunakan **Elbow Method**.

Berdasarkan hasil analisis diperoleh:

### `K = 2`

Sehingga data dikelompokkan menjadi dua kategori:

| Cluster | Kategori | Interpretasi |
|---|---|---|
| 🔴 Cluster 0 | **Prioritas Tinggi** | Daerah dengan kinerja pengelolaan sampah relatif lebih rendah sehingga memerlukan perhatian lebih lanjut |
| 🟢 Cluster 1 | **Prioritas Rendah** | Daerah dengan kinerja pengelolaan sampah relatif lebih baik |

Label cluster ditentukan berdasarkan analisis karakteristik dan nilai centroid dari masing-masing cluster.

---

## 📈 Evaluasi Clustering

Hasil clustering dievaluasi menggunakan **Silhouette Score**.

| Metrik | Hasil |
|---|---:|
| Jumlah Cluster | **2** |
| Silhouette Score | **0.671784** |

Silhouette Score memiliki rentang nilai dari **-1 hingga 1**.

Nilai yang semakin mendekati **1** menunjukkan bahwa objek dalam suatu cluster memiliki tingkat kemiripan yang tinggi dan memiliki pemisahan yang baik terhadap cluster lainnya.

Nilai **0.671784** menunjukkan bahwa hasil clustering memiliki struktur dan pemisahan cluster yang cukup baik.

---

## 🗺️ Mapping Hasil Clustering

Hasil clustering divisualisasikan menggunakan **peta interaktif Kalimantan Timur** untuk menunjukkan persebaran daerah berdasarkan tingkat prioritas.

**Keterangan:**

🔴 **Merah — Prioritas Tinggi**

🟢 **Hijau — Prioritas Rendah**

Visualisasi geografis membantu pengguna memahami persebaran kinerja pengelolaan sampah antar kabupaten/kota secara lebih intuitif.

---

## 💡 Insight

Hasil analisis menunjukkan adanya perbedaan karakteristik kinerja pengelolaan sampah antar kabupaten/kota di Kalimantan Timur.

Daerah pada **Cluster Prioritas Tinggi** memiliki capaian pengelolaan sampah yang relatif lebih rendah sehingga dapat menjadi perhatian dalam peningkatan strategi pengurangan dan penanganan sampah.

Sementara itu, daerah pada **Cluster Prioritas Rendah** menunjukkan capaian pengelolaan sampah yang relatif lebih baik berdasarkan indikator yang dianalisis.

Hasil clustering dapat digunakan sebagai informasi pendukung dalam melakukan evaluasi dan pemetaan kondisi pengelolaan sampah di Kalimantan Timur.

---

## 🚀 Deployment

Hasil analisis dikembangkan menjadi aplikasi web interaktif menggunakan **Streamlit**.

### 🌐 Live Dashboard

👉 [**Buka Dashboard Kinerja Pengelolaan Sampah Kaltim**](https://kinerja-pengelolaan-sampah-kaltim.streamlit.app/)

Dashboard menyediakan beberapa fitur:

- 📊 Analisis data pengelolaan sampah
- 🤖 Hasil K-Means Clustering
- 🗺️ Mapping hasil cluster
- 📋 Dataset
- ℹ️ Informasi proyek

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Algorithm | K-Means |
| Preprocessing | MinMaxScaler |
| Evaluation | Silhouette Score |
| Visualization | Matplotlib, Seaborn, Plotly |
| Mapping | Folium |
| Deployment | Streamlit |
| Development | Jupyter Notebook |
| Version Control | Git & GitHub |

---

## 📁 Struktur Repository

```text
Kinerja-Pengelolaan-Sampah-Kaltim/
│
├── data/
│   └── dataset.csv
│
├── notebook/
│   └── KMeans_Clustering.ipynb
│
├── images/
│   ├── dashboard.png
│   ├── elbow_method.png
│   ├── clustering.png
│   └── mapping.png
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📸 Dashboard Preview

<p align="center">
  <img src="images/dashboard.png" width="850">
</p>

### Hasil Clustering

<p align="center">
  <img src="images/clustering.png" width="750">
</p>

### Mapping Cluster

<p align="center">
  <img src="images/mapping.png" width="750">
</p>

---

## 💼 Manfaat

Proyek ini diharapkan dapat:

- Memberikan gambaran mengenai kinerja pengelolaan sampah di Kalimantan Timur.
- Mengidentifikasi daerah yang membutuhkan perhatian lebih lanjut.
- Mempermudah perbandingan kinerja antar kabupaten/kota.
- Mendukung evaluasi pengelolaan sampah menggunakan pendekatan berbasis data.
- Menunjukkan penerapan **Machine Learning** pada permasalahan lingkungan.

---

## 🔮 Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan:

- Memperbarui dataset dengan data tahun terbaru.
- Membandingkan K-Means dengan algoritma clustering lainnya.
- Menambahkan variabel kependudukan dan sosial ekonomi.
- Mengembangkan analisis tren (*time series*).
- Mengintegrasikan data secara otomatis melalui API.
- Mengembangkan dashboard monitoring secara berkala.

---

## 👨‍💻 Author

**Andi Muhammad Rivaldy Aljidannur**

Lulusan Sistem Informasi | Data Analyst | Data Science Enthusiast

### Skills

`Python` • `SQL` • `Machine Learning` • `Data Analysis` • `Power BI` • `Streamlit`

---

<p align="center">
  <b>⭐ Jika project ini menarik, jangan lupa berikan Star pada repository ini.</b>
</p>

<p align="center">
  ♻️ <b>Data untuk Pengelolaan Lingkungan yang Lebih Baik</b>
</p>
