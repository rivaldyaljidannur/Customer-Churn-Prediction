import streamlit as st
import pandas as pd
import numpy as np
import io
import model_utils

# ==================== PAGE CONFIG & STYLING ====================
st.set_page_config(
    page_title="Telco Customer Churn Prediction System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #F8FAFC;
    }
    .stAppHeader {
        background-color: transparent;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 50%, #3B82F6 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }
    .header-banner h1 {
        color: #FFFFFF;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .header-banner p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.25rem;
    }
    .metric-desc {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 0.25rem;
    }
    
    /* Content Cards */
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Badge styling */
    .badge-primary {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Risk meter cards */
    .risk-card-high {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .risk-card-medium {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .risk-card-low {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading & training
@st.cache_resource(show_spinner="Sedang melatih model Random Forest Classifier...")
def get_model_artifacts():
    return model_utils.train_and_evaluate()

artifacts = get_model_artifacts()
model = artifacts['model']
metrics = artifacts['metrics']
feature_names = artifacts['feature_names']
df_raw = artifacts['df_raw']
df_prep = artifacts['df_prep']
importance_df = artifacts['importance_df']
cm = artifacts['cm']
fpr = artifacts['fpr']
tpr = artifacts['tpr']
roc_auc = artifacts['roc_auc']
report_df = artifacts['report_df']

# ==================== SIDEBAR NAVIGATION ====================
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/data-configuration.png", width=70)
    st.title("Telco Analytics")
    st.caption("Random Forest Churn Prediction System")
    st.markdown("---")
    
    menu = st.radio(
        "📍 **Navigasi Modul**",
        [
            "📖 1. Deskripsi & Metodologi",
            "🔍 2. Exploratory Data Analysis (EDA)",
            "📊 3. Performa & Evaluasi Model",
            "🌳 4. Feature Importance & Insight",
            "🎯 5. Simulasi Prediksi Interactive",
            "💡 6. Insight & Strategi Bisnis"
        ]
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Metadata Model")
    st.markdown(f"**Algoritma:** Random Forest")
    st.markdown(f"**Estimators:** 100 Trees")
    st.markdown(f"**Stratified Split:** 80% Train / 20% Test")
    st.markdown(f"**Accuracy:** `{metrics['Accuracy']*100:.2f}%`")
    st.markdown(f"**ROC-AUC:** `{metrics['ROC-AUC']:.4f}`")
    st.markdown("---")
    st.caption("Developed with Streamlit & Scikit-Learn | CRISP-DM Framework")

# ==================== HEADER BANNER ====================
st.markdown("""
<div class="header-banner">
    <h1>📡 Telco Customer Churn Prediction System</h1>
    <p>Sistem Prediksi Risiko Customer Churn Berbasis Machine Learning (Random Forest & CRISP-DM)</p>
</div>
""", unsafe_allow_html=True)


# ==================== MODUL 1: DESKRIPSI & METODOLOGI ====================
if menu == "📖 1. Deskripsi & Metodologi":
    st.markdown("## 📖 Deskripsi Proyek & Metodologi CRISP-DM")
    
    st.markdown("""
    <div class="card">
        <h3>📌 Latar Belakang Bisnis</h3>
        <p style="text-align: justify; font-size: 1.05rem; line-height: 1.6; color: #334155;">
            <b>Customer churn</b> merupakan kondisi ketika pelanggan memutuskan untuk berhenti menggunakan layanan perusahaan telekomunikasi. 
            Tingginya tingkat customer churn berdampak langsung pada penurunan pendapatan bulanan serta meningkatnya biaya akuisisi pelanggan baru (Customer Acquisition Cost). 
            Oleh karena itu, perusahaan memerlukan sistem Machine Learning yang presisi untuk mendeteksi potensi churn lebih awal agar strategi retensi dapat dijalankan secara proaktif.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🎯 Tujuan Proyek</h3>
            <ul style="font-size: 1rem; line-height: 1.7; color: #334155;">
                <li><b>Menganalisis karakteristik</b> & pola perilaku pelanggan berdasarkan data historis Telco.</li>
                <li><b>Mengidentifikasi faktor-faktor kunci</b> yang paling memengaruhi keputusan churn pelanggan.</li>
                <li><b>Melakukan persiapan data (Data Preparation)</b> termasuk penanganan missing value, duplikasi, dan encoding fitur.</li>
                <li><b>Membangun model presisi tinggi</b> menggunakan algoritma <b>Random Forest Classifier</b>.</li>
                <li><b>Mengevaluasi performa model</b> dengan metrik Accuracy, Precision, Recall, F1-Score, dan ROC-AUC.</li>
                <li><b>Menyediakan alat simulasi prediksi interaktif</b> (Single & Batch) untuk mendukung <i>Data-Driven Decision Making</i>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🔄 Metodologi CRISP-DM</h3>
            <ol style="font-size: 0.95rem; line-height: 1.6; color: #334155;">
                <li><b>Business Understanding:</b> Menentukan tujuan bisnis retensi & metrik evaluasi target.</li>
                <li><b>Data Understanding:</b> Eksplorasi struktur data (7.043 baris, 21 atribut).</li>
                <li><b>Data Preparation:</b> Pembersihan <i>TotalCharges</i>, hapus duplikat, & One-Hot Encoding.</li>
                <li><b>Modeling:</b> Pelatihan model ensemble Random Forest dengan 100 decision trees.</li>
                <li><b>Evaluation:</b> Pengujian model pada 20% data test (Stratified Split).</li>
                <li><b>Deployment:</b> Implementasi dashboard interaktif Streamlit untuk operasional bisnis.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📂 Atribut Dataset Telco Customer Churn")
    
    t1, t2, t3, t4 = st.tabs(["👤 Demografi", "📱 Informasi Layanan", "💳 Kontrak & Pembayaran", "🎯 Target Variables"])
    
    with t1:
        st.dataframe(pd.DataFrame({
            "Fitur": ["Gender", "SeniorCitizen", "Partner", "Dependents"],
            "Tipe Data": ["Kategorikal", "Biner (0/1)", "Kategorikal (Yes/No)", "Kategorikal (Yes/No)"],
            "Keterangan": ["Jenis kelamin pelanggan", "Apakah pelanggan lansia (>65 thn)", "Apakah memiliki pasangan", "Apakah memiliki tanggungan keluarga"]
        }), use_container_width=True)
        
    with t2:
        st.dataframe(pd.DataFrame({
            "Fitur": ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"],
            "Keterangan": ["Layanan telepon utama", "Memiliki banyak saluran telepon", "Penyedia internet (DSL, Fiber optic, No)", "Layanan keamanan cyber", "Layanan cadangan cloud", "Perlindungan perangkat", "Dukungan teknis prioritas", "Layanan streaming TV", "Layanan streaming film"]
        }), use_container_width=True)
        
    with t3:
        st.dataframe(pd.DataFrame({
            "Fitur": ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "tenure"],
            "Keterangan": ["Jenis kontrak (Month-to-month, 1 Year, 2 Year)", "Tagihan elektronik (Yes/No)", "Metode pembayaran (E-check, Mailed check, Bank, Credit card)", "Biaya bulanan (USD)", "Total akumulasi biaya (USD)", "Lama berlangganan dalam bulan"]
        }), use_container_width=True)
        
    with t4:
        st.dataframe(pd.DataFrame({
            "Fitur": ["Churn"],
            "Status": ["Yes / No"],
            "Keterangan": ["Yes → Pelanggan berhenti berlangganan (Churn) | No → Pelanggan tetap berlangganan (Retained)"]
        }), use_container_width=True)


# ==================== MODUL 2: EXPLORATORY DATA ANALYSIS (EDA) ====================
elif menu == "🔍 2. Exploratory Data Analysis (EDA)":
    st.markdown("## 🔍 Exploratory Data Analysis (EDA)")
    st.caption("Eksplorasi mendalam untuk memahami pola data historis pelanggan telekomunikasi.")
    
    # Overview Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Samples</div>
            <div class="metric-value">{len(df_raw):,}</div>
            <div class="metric-desc">Baris Pelanggan</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        churn_count = (df_raw['Churn'] == 'Yes').sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Churn Customers</div>
            <div class="metric-value" style="color: #EF4444;">{churn_count:,}</div>
            <div class="metric-desc">{churn_count/len(df_raw)*100:.1f}% Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        retained_count = (df_raw['Churn'] == 'No').sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Retained Customers</div>
            <div class="metric-value" style="color: #10B981;">{retained_count:,}</div>
            <div class="metric-desc">{retained_count/len(df_raw)*100:.1f}% Retained</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Features</div>
            <div class="metric-value">{df_raw.shape[1] - 1}</div>
            <div class="metric-desc">Atribut Prediktor</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_eda1, tab_eda2, tab_eda3, tab_eda4 = st.tabs([
        "📊 Ringkasan & Faktor Utama", 
        "💳 Biaya & Pembayaran", 
        "🛡️ Layanan & Produk", 
        "📋 Data Explorer"
    ])
    
    with tab_eda1:
        st.markdown("### 🎯 Ringkasan Status Churn & Faktor Utama")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            fig_churn = model_utils.plot_churn_distribution(df_raw)
            st.plotly_chart(fig_churn, use_container_width=True)
        with col_b:
            fig_contract = model_utils.plot_churn_by_contract(df_raw)
            st.plotly_chart(fig_contract, use_container_width=True)
            
        st.markdown("---")
        fig_tenure = model_utils.plot_churn_by_tenure_group(df_raw)
        st.plotly_chart(fig_tenure, use_container_width=True)
        
        st.markdown("""
        <div class="card">
            <h4>💡 Insight Ringkasan Utama</h4>
            <ul>
                <li><b>Kontrak Month-to-Month</b> memiliki churn rate tertinggi mencapai <b>42.7%</b>, sementara Kontrak 2 Tahun hanya <b>2.8%</b>.</li>
                <li><b>Masa Berlangganan (Tenure):</b> Risiko churn sangat kritis pada tahun pertama (<b>47.7%</b>) dan menurun drastis menjadi di bawah <b>7%</b> setelah 5 tahun.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
            
    with tab_eda2:
        st.markdown("### 💳 Analisis Biaya Bulanan & Risiko Metode Pembayaran")
        col_c, col_d = st.columns([1, 1])
        with col_c:
            fig_charge = model_utils.plot_churn_by_monthly_charges_group(df_raw)
            st.plotly_chart(fig_charge, use_container_width=True)
        with col_d:
            fig_pay = model_utils.plot_churn_by_payment_method(df_raw)
            st.plotly_chart(fig_pay, use_container_width=True)
            
        st.markdown("""
        <div class="card">
            <h4>💡 Insight Biaya & Pembayaran</h4>
            <ul>
                <li><b>Biaya Bulanan Tinggi:</b> Pelanggan dengan tagihan bulanan > $70/bulan memiliki churn rate tertinggi (hingga <b>37.5%</b>).</li>
                <li><b>Metode Pembayaran:</b> Pengguna <b>Electronic Check</b> berisiko churn paling tinggi (<b>45.3%</b>), jauh melebihi metode Auto-Debit Bank & Kartu Kredit.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_eda3:
        st.markdown("### 🛡️ Dampak Layanan Tambahan & Jenis Internet")
        col_e, col_f = st.columns([1, 1])
        with col_e:
            fig_serv = model_utils.plot_services_impact(df_raw)
            st.plotly_chart(fig_serv, use_container_width=True)
        with col_f:
            fig_net = model_utils.plot_internet_service_churn(df_raw)
            st.plotly_chart(fig_net, use_container_width=True)
            
        st.markdown("""
        <div class="card">
            <h4>💡 Insight Ekosistem Layanan</h4>
            <ul>
                <li><b>Layanan Keamanan & Tech Support:</b> Pelanggan yang TIDAK memiliki Tech Support atau Online Security berisiko churn hingga <b>41.7%</b>, dibanding yang memiliki fitur tersebut (< <b>15%</b>).</li>
                <li><b>Internet Fiber Optic:</b> Pengguna Fiber Optic memiliki churn rate <b>41.9%</b>, mengindikasikan sensitivitas tinggi terhadap harga dan ekspektasi kualitas koneksi.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_eda4:
        st.subheader("📋 Dataset Telco Churn Raw Viewer")
        st.caption("Gunakan tabel interaktif di bawah ini untuk melihat dan memfilter data historis pelanggan.")
        st.dataframe(df_raw, use_container_width=True)


# ==================== MODUL 3: PERFORMA & EVALUASI MODEL ====================
elif menu == "📊 3. Performa & Evaluasi Model":
    st.markdown("## 📊 Performa & Evaluasi Model Random Forest")
    st.caption("Hasil pengujian model pada 20% data test (Stratified Split).")
    
    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value" style="color: #2563EB;">{metrics['Accuracy']*100:.2f}%</div>
            <div class="metric-desc">Tingkat Ketepatan Total</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-value" style="color: #0D9488;">{metrics['Precision']*100:.2f}%</div>
            <div class="metric-desc">Ketepatan Prediksi Churn</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-value" style="color: #D97706;">{metrics['Recall']*100:.2f}%</div>
            <div class="metric-desc">Sensitivitas Deteksi Churn</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">F1-Score</div>
            <div class="metric-value" style="color: #9333EA;">{metrics['F1-Score']:.4f}</div>
            <div class="metric-desc">Harmonik Precision & Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ROC-AUC</div>
            <div class="metric-value" style="color: #059669;">{metrics['ROC-AUC']:.4f}</div>
            <div class="metric-desc">Kemampuan Separasi Model</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        fig_cm = model_utils.plot_confusion_matrix(cm)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_eval2:
        fig_roc = model_utils.plot_roc_curve(fpr, tpr, roc_auc)
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("### 📝 Classification Report Detil")
    st.dataframe(report_df, use_container_width=True)


# ==================== MODUL 4: FEATURE IMPORTANCE ====================
elif menu == "🌳 4. Feature Importance & Insight":
    st.markdown("## 🌳 Feature Importance Model Random Forest")
    st.caption("Peringkat variabel prediktor yang paling memengaruhi keputusan prediksi churn.")
    
    top_n = st.slider("Tampilkan Top N Fitur Teratas:", min_value=5, max_value=25, value=12)
    fig_imp = model_utils.plot_feature_importance(importance_df, top_n=top_n)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("""
    <div class="card">
        <h3>🔍 Penjelasan Faktor-Faktor Kunci Churn</h3>
        <ol style="font-size: 1rem; line-height: 1.7; color: #334155;">
            <li><b>Tenure (Masa Berlangganan):</b> Merupakan variabel paling dominan. Semakin lama pelanggan berada bersama perusahaan, semakin kecil risiko mereka berpindah.</li>
            <li><b>Monthly Charges & Total Charges:</b> Sensitivitas harga memainkan peran penting. Biaya bulanan tinggi memicu pelanggan mencari alternatif layanan yang lebih terjangkau.</li>
            <li><b>Contract_Two year & Contract_One year:</b> Jenis kontrak berpengaruh signifikan. Kontrak jangka panjang secara efektif mengikat dan mengunci komitmen pelanggan.</li>
            <li><b>InternetService_Fiber optic:</b> Pengguna paket Fiber Optic memiliki tingkat keluhan/pindah yang lebih tinggi dibanding DSL, mengindikasikan ekspektasi tinggi terhadap stabilitas jaringan Fiber Optic.</li>
            <li><b>TechSupport_No & OnlineSecurity_No:</b> Ketiadaan fitur keamanan & dukungan teknis meningkatkan kerapuhan loyalitas pelanggan.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ==================== MODUL 5: SIMULASI PREDIKSI INTERACTIVE ====================
elif menu == "🎯 5. Simulasi Prediksi Interactive":
    st.markdown("## 🎯 Simulasi Prediksi Customer Churn")
    
    tab_sim1, tab_sim2 = st.tabs(["👤 Input Prediksi Individu", "📁 Batch Upload Prediksi (CSV/Excel)"])
    
    with tab_sim1:
        st.markdown("### 📝 Form Parameter Pelanggan")
        st.caption("Masukkan detail profil pelanggan untuk mendapatkan estimasi probabilitas churn real-time.")
        
        with st.form("single_predict_form"):
            c_sec1, c_sec2, c_sec3 = st.columns(3)
            
            with c_sec1:
                st.subheader("1. Profil Demografi")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen (>65 thn)", [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
                partner = st.selectbox("Memiliki Partner", ["Yes", "No"])
                dependents = st.selectbox("Memiliki Tanggungan", ["Yes", "No"])
                tenure = st.number_input("Tenure (Bulan Berlangganan)", min_value=0, max_value=72, value=12)
                
            with c_sec2:
                st.subheader("2. Layanan Telekomunikasi")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                
            with c_sec3:
                st.subheader("3. Kontrak & Biaya")
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=75.0)
                total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(tenure * monthly_charges))
                
            submit_btn = st.form_submit_button("🚀 Jalankan Prediksi Churn", use_container_width=True)
            
        if submit_btn:
            input_dict = {
                'gender': gender,
                'SeniorCitizen': senior,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            res = model_utils.predict_single_customer(model, feature_names, input_dict)
            
            st.markdown("---")
            st.markdown("### 📊 Hasil Analisis Prediksi")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                prob_pct = res['probability'] * 100
                st.markdown(f"#### Probabilitas Churn: `{prob_pct:.1f}%`")
                st.progress(res['probability'])
                
                if res['prediction'] == 'Yes':
                    st.error(f"⚠️ **STATUS PREDIKSI: BERPOTENSI CHURN (YES)**\nTingkat Risiko: {res['risk_level']}")
                else:
                    st.success(f"✅ **STATUS PREDIKSI: CENDERUNG RETAINED (NO)**\nTingkat Risiko: {res['risk_level']}")

            with res_col2:
                st.markdown("#### 💡 Rekomendasi Retensi Terpersonalisasi:")
                for rec in res['recommendations']:
                    st.markdown(rec)
                    
    with tab_sim2:
        st.markdown("### 📁 Batch Prediction Tool")
        st.caption("Unggah file CSV atau Excel yang berisi sekumpulan data pelanggan untuk melakukan prediksi massal.")
        
        uploaded_file = st.file_uploader("Unggah File (CSV / XLSX):", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    batch_df = pd.read_csv(uploaded_file)
                else:
                    batch_df = pd.read_excel(uploaded_file)
                    
                st.success(f"File `{uploaded_file.name}` berhasil diunggah! (Total {len(batch_df)} baris)")
                st.dataframe(batch_df.head(5), use_container_width=True)
                
                if st.button("⚡ Proses Prediksi Batch"):
                    with st.spinner("Memproses prediksi..."):
                        batch_res = model_utils.predict_batch(model, feature_names, batch_df)
                        
                        st.subheader("🎉 Hasil Prediksi Batch")
                        st.dataframe(batch_res, use_container_width=True)
                        
                        # Summary stats
                        b_churn = (batch_res['Predicted_Churn'] == 'Yes').sum()
                        b_total = len(batch_res)
                        st.info(f"📊 Ringkasan: **{b_churn} dari {b_total} pelanggan ({b_churn/b_total*100:.1f}%)** diprediksi berisiko Churn.")
                        
                        # Download Button
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            batch_res.to_excel(writer, index=False, sheet_name='Hasil_Prediksi')
                        buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Unduh Hasil Prediksi (Excel .xlsx)",
                            data=buffer,
                            file_name="Hasil_Prediksi_Churn_Batch.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")


# ==================== MODUL 6: INSIGHT & STRATEGI BISNIS ====================
elif menu == "💡 6. Insight & Strategi Bisnis":
    st.markdown("## 💡 Insight Bisnis & Strategi Retensi Pelanggan")
    st.caption("Rekomendasi taktis berbasis hasil pemodelan Machine Learning Random Forest.")
    
    c_ins1, c_ins2 = st.columns(2)
    
    with c_ins1:
        st.markdown("""
        <div class="card">
            <h3>📈 Strategi 1: Migrasi Kontrak Jangka Panjang</h3>
            <p><b>Temuan:</b> Pelanggan dengan kontrak Month-to-month memiliki proporsi churn tertinggi.</p>
            <p><b>Aksi Taktis:</b></p>
            <ul>
                <li>Berikan program insentif (diskon biaya bulanan 10-15%) bagi pelanggan yang mau melakukan upgrade dari kontrak bulanan ke kontrak 1 atau 2 tahun.</li>
                <li>Gunakan reminder otomatis sebelum kontrak bulanan diperbarui untuk menawarkan paket loyalitas.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>🛡️ Strategi 2: Bundling Layanan Keamanan & Tech Support</h3>
            <p><b>Temuan:</b> Pelanggan tanpa Tech Support dan Online Security jauh lebih mudah churn.</p>
            <p><b>Aksi Taktis:</b></p>
            <ul>
                <li>Berikan uji coba gratis (Free Trial 3-6 bulan) add-on <i>Online Security</i> dan <i>Tech Support</i> untuk pelanggan baru pengguna Fiber Optic.</li>
                <li>Buat paket penawaran 'All-in-One Security Bundle' dengan harga yang lebih terjangkau.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c_ins2:
        st.markdown("""
        <div class="card">
            <h3>🎁 Strategi 3: Program Onboarding & Early Tenure Loyalty</h3>
            <p><b>Temuan:</b> Tingkat churn paling kritis terjadi pada 12 bulan pertama (tenure rendah).</p>
            <p><b>Aksi Taktis:</b></p>
            <ul>
                <li>Bangun <i>First-Year Onboarding Journey</i> dengan touchpoint di bulan ke-1, 3, 6, dan 12 untuk memastikan kepuasan jaringan.</li>
                <li>Berikan hadiah loyalitas (cashback/bonus kuota) di akhir bulan ke-6 dan bulan ke-12.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>💳 Strategi 4: Optimalisasi Metode Pembayaran</h3>
            <p><b>Temuan:</b> Pembayaran via Electronic Check memiliki rasio churn tertinggi.</p>
            <p><b>Aksi Taktis:</b></p>
            <ul>
                <li>Dorong pelanggan melakukan migrasi dari Electronic Check ke Auto-Debit (Bank Transfer / Credit Card) dengan memberikan potongan tagihan satu kali (one-time discount).</li>
                <li>Tingkatkan keandalan sistem pembayaran tagihan elektronik agar tidak memicu kegagalan transaksi yang mengganggu pelanggan.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748B;'>© 2026 Customer Churn Prediction System | Powered by Streamlit & Random Forest</p>", unsafe_allow_html=True)
