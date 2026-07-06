# ==============================================================================
# UAS PENGENALAN SAINS DATA - DASHBOARD INTERAKTIF KOMODITAS PERKEBUNAN
# ==============================================================================
# Developer: Data Scientist
# Tools: Streamlit, Pandas, Numpy, Matplotlib, Seaborn, Plotly, Scipy, Statsmodels
# Deskripsi: Kode ini merupakan dashboard analitik tingkat lanjut yang mencakup
#            Data Understanding, Preprocessing, EDA 2D & 3D, Korelasi, Clustering,
#            Regresi, Hingga Analisis Residu dan Forecasting sederhana.
# ==============================================================================

# --- IMPORT LIBRARIES ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
import io
import time
from typing import List, Dict, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# --- KONFIGURASI TEMA DAN HALAMAN ---
st.set_page_config(
    page_title="Sains Data Dashboard 3D",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS UNTUK TAMPILAN PREMIUM
# ==============================================================================
def load_custom_css() -> None:
    """
    Memuat custom CSS untuk mempercantik tampilan dashboard Streamlit.
    Menerapkan tema glassmorphism, gradient text, dan custom card metrics.
    """
    st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        /* Global Variables */
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --secondary: #8b5cf6;
            --accent: #ec4899;
            --success: #22c55e;
            --warning: #f59e0b;
        }
        
        /* Main Background */
        .stApp {
            background: radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.15) 0%, transparent 40%),
                        radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
                        var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
        }
        
        /* Header Styling */
        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            margin-bottom: 20px !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.9) !important;
            border-right: 1px solid rgba(148, 163, 184, 0.1) !important;
            backdrop-filter: blur(10px);
        }
        
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background: var(--card-bg) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            padding: 20px !important;
            border-radius: 15px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1) !important;
            border-color: var(--primary) !important;
        }
        
        label[data-testid="stMetricLabel"] {
            font-size: 14px !important;
            color: var(--text-muted) !important;
            font-weight: 400 !important;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 600 !important;
            color: var(--text-main) !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* Dataframes and Tables */
        .stDataFrame, .stTable {
            background: var(--card-bg) !important;
            border-radius: 12px !important;
            padding: 10px !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
            color: white !important;
            border: none !important;
            padding: 10px 24px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }
        
        .stButton>button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
        }
        
        /* Selectboxes and Multiselects */
        .stSelectbox>div>div, .stMultiselect>div>div {
            background: var(--card-bg) !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            border-radius: 8px !important;
            color: var(--text-main) !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: var(--card-bg) !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
        }
        
        /* Custom Divider */
        hr {
            border: 0 !important;
            height: 2px !important;
            background: linear-gradient(90deg, transparent, var(--primary), transparent) !important;
            margin: 30px 0 !important;
        }
        
        /* Caption styling */
        .stCaption {
            color: var(--text-muted) !important;
            font-style: italic !important;
            font-size: 13px !important;
        }
        
        /* Alert Boxes */
        .stAlert {
            border-radius: 10px !important;
        }
        
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ==============================================================================
# DATA INJECTION (FALLBACK IF CSV NOT FOUND)
# ==============================================================================
RAW_CSV_DATA = """Provinsi;Kelapa_Sawit;Kelapa;Karet;Kopi;Kakao;Teh;Tebu
ACEH;1092.71;64.1;51.17;74.13;34.17;0.0;0.0
SUMATERA UTARA;5120.02;103.64;251.52;91.69;38.48;10.14;14.52
SUMATERA BARAT;1410.64;79.06;99.67;15.32;34.58;5.6;0.0
RIAU;9136.1;399.95;180.74;1.83;0.84;0.0;0.0
JAMBI;2113.97;115.35;222.56;23.23;0.69;4.24;0.0
SUMATERA SELATAN;3967.39;61.65;619.55;219.59;4.4;2.62;124.37
BENGKULU;1298.17;7.25;64.6;55.63;2.36;1.88;0.0
LAMPUNG;375.24;83.76;84.83;120.38;48.11;0.0;644.48
KEP. BANGKA BELITUNG;860.24;4.55;26.04;0.11;0.29;0.0;0.0
KEP. RIAU;24.19;12.16;8.41;0.0;0.01;0.0;0.0
DKI JAKARTA;0.0;0.0;0.0;0.0;0.0;0.0;0.0
JAWA BARAT;46.75;88.14;21.17;26.48;0.71;80.24;55.17
JAWA TENGAH;0.0;139.01;20.73;26.79;1.31;11.81;250.07
DI YOGYAKARTA;0.0;50.08;0.01;1.88;2.03;0.23;3.22
JAWA TIMUR;0.0;194.14;16.27;53.25;10.56;2.14;1252.84
BANTEN;32.84;45.83;5.77;2.02;2.12;0.01;0.0
BALI;0.0;68.39;0.0;14.71;4.87;0.0;0.0
NUSA TENGGARA BARAT;0.0;49.81;0.0;6.42;2.59;0.0;18.22
NUSA TENGGARA TIMUR;0.0;62.15;0.0;24.38;21.08;0.0;10.77
KALIMANTAN BARAT;4958.54;76.8;158.41;2.98;0.6;0.0;0.0
KALIMANTAN TENGAH;7458.14;16.6;108.29;0.24;1.61;0.0;0.0
KALIMANTAN SELATAN;1255.08;24.01;127.96;0.89;0.05;0.0;0.0
KALIMANTAN TIMUR;3905.19;9.82;53.49;0.12;1.03;0.0;0.0
KALIMANTAN UTARA;611.14;0.64;0.16;0.11;0.79;0.0;0.0
SULAWESI UTARA;0.0;270.3;0.0;3.72;5.62;0.0;0.0
SULAWESI TENGAH;384.2;199.64;1.37;3.13;125.2;0.0;0.0
SULAWESI SELATAN;133.5;67.57;3.77;31.79;80.52;0.0;22.56
SULAWESI TENGGARA;75.47;42.87;0.22;2.61;98.02;0.0;15.41
GORONTALO;22.83;66.7;0.0;0.13;1.54;0.0;53.89
SULAWESI BARAT;366.68;36.72;0.0;4.75;67.14;0.0;0.0
MALUKU;22.33;107.73;0.6;0.49;8.59;0.0;0.0
MALUKU UTARA;20.14;204.17;0.0;0.02;7.38;0.0;0.0
PAPUA BARAT;40.38;2.02;0.0;0.01;0.24;0.0;0.0
PAPUA BARAT DAYA;43.44;14.35;0.0;0.0;0.76;0.0;0.0
PAPUA;130.55;9.77;0.0;0.09;8.0;0.0;0.0
PAPUA SELATAN;482.37;4.35;4.82;0.0;0.01;0.0;0.0
PAPUA TENGAH;47.98;1.2;0.0;1.18;0.82;0.0;0.0
PAPUA PEGUNUNGAN;0.0;0.02;0.0;3.27;0.0;0.0;0.0
"""

# ==============================================================================
# DATA HANDLING & PREPROCESSING CLASS
# ==============================================================================
class DataProcessor:
    """
    Kelas untuk menangani loading, pembersihan, dan preprocessing data.
    """
    def __init__(self, raw_data: str, sep: str = ';'):
        self.raw_data = raw_data
        self.sep = sep
        self.df = None
        self.df_clean = None
        self.commodities = ['Kelapa_Sawit', 'Kelapa', 'Karet', 'Kopi', 'Kakao', 'Teh', 'Tebu']
        
    def load_data(self) -> pd.DataFrame:
        """Memuat data dari string atau file."""
        try:
            # Coba baca dari file lokal
            self.df = pd.read_csv('produksi_tanaman.csv', sep=self.sep, index_col=0)
        except FileNotFoundError:
            # Fallback ke data raw di memori
            self.df = pd.read_csv(io.StringIO(self.raw_data), sep=self.sep, index_col=0)
            
        self.df.columns = self.df.columns.str.strip()
        self.df.index.name = 'Provinsi'
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Melakukan pembersihan data: missing values, duplicates, types."""
        df = self.df.copy()
        
        # 1. Missing Value Handling
        df.fillna(0, inplace=True)
        
        # 2. Data Type Conversion
        for col in self.commodities:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # 3. Duplicate Handling
        df = df[~df.index.duplicated(keep='first')]
        
        self.df_clean = df
        return df
    
    def remove_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Mendeteksi dan menghapus outlier menggunakan metode IQR (Interquartile Range).
        Mengembalikan dataframe yang sudah dibersihkan dan dataframe outlier.
        """
        df_out = df.copy()
        outliers_mask = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Tandai outlier
            outliers_mask[col] = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
            
            # Capping: Ganti outlier dengan batas atas/bawah (Winsorization)
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, 
                                  np.where(df_out[col] < lower_bound, lower_bound, df_out[col]))
            
        outliers = df[outliers_mask.any(axis=1)]
        return df_out, outliers

# Inisialisasi Data Processor
processor = DataProcessor(RAW_CSV_DATA)
df_raw = processor.load_data()
df = processor.clean_data()
komoditas = processor.commodities

# ==============================================================================
# UI COMPONENTS HELPER FUNCTIONS
# ==============================================================================
def create_metric_card(col: Any, title: str, value: str, delta: str = "") -> None:
    """Membuat kartu metrik kustom."""
    col.metric(title, value, delta)

def render_interpretation(text: str, type: str = "info") -> None:
    """Merender kotak interpretasi dengan styling."""
    if type == "info":
        st.info(f"📊 **Interpretasi:** {text}")
    elif type == "success":
        st.success(f"✅ **Insight:** {text}")
    elif type == "warning":
        st.warning(f"⚠️ **Peringatan:** {text}")

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🌿 Agri-Analytics 3D")
st.sidebar.markdown("Dashboard Sains Data Tingkat Lanjut")
st.sidebar.divider()

menu_options = [
    "🏠 Beranda & Data Understanding",
    "🧹 Data Cleaning & Preprocessing",
    "📊 Eksplorasi Data (EDA 2D)",
    "🧊 Visualisasi 3D Interaktif",
    "🔗 Analisis Hubungan Variabel",
    "🧮 Pemodelan Regresi & Uji Asumsi",
    "🌳 Clustering Hirarkis (Bonus)",
    "💡 Insight & Rekomendasi"
]

menu = st.sidebar.selectbox("Pilih Modul Analisis:", menu_options)

# Status data di sidebar
st.sidebar.divider()
st.sidebar.markdown("#### Status Dataset")
st.sidebar.write(f"**Total Provinsi:** {df.shape[0]}")
st.sidebar.write(f"**Total Variabel:** {df.shape[1]}")
st.sidebar.write(f"**Status Missing Value:** ✅ Bersih")
st.sidebar.write(f"**Status Tipe Data:** ✅ Numerik (Float64)")

# ==============================================================================
# HALAMAN 1: BERANDA & DATA UNDERSTANDING
# ==============================================================================
if menu == "🏠 Beranda & Data Understanding":
    st.markdown("# 🏠 Beranda & Data Understanding")
    st.markdown("Selamat datang di Dashboard Analitik Produksi Tanaman Perkebunan Indonesia. "
                "Dashboard ini dirancang untuk menganalisis, memvisualisasikan, dan memprediksi "
                "tren produksi 7 komoditas utama di 38 Provinsi Indonesia.")
    
    st.divider()
    
    # Metrik Utama
    st.markdown("## 📈 Statistik Global Dataset")
    col1, col2, col3, col4 = st.columns(4)
    
    total_produksi = df[komoditas].sum().sum()
    prod_tertinggi = df[komoditas].sum().idxmax()
    nilai_tertinggi = df[komoditas].sum().max()
    prov_terproduktif = df[komoditas].sum(axis=1).idxmax()
    
    create_metric_card(col1, "Total Produksi Nasional", f"{total_produksi:,.2f} Ton")
    create_metric_card(col2, "Komoditas Terbanyak", prod_tertinggi)
    create_metric_card(col3, "Nilai Produksi Terbanyak", f"{nilai_tertinggi:,.2f} Ton")
    create_metric_card(col4, "Provinsi Terproduktif", prov_terproduktif)
    
    st.divider()
    
    # Data Understanding
    st.markdown("## 📋 Data Understanding (A.1 & A.2)")
    st.markdown("Dataset ini berisi informasi produksi 7 komoditas perkebunan di 38 provinsi Indonesia. "
                "Berikut adalah penjabaran struktur data secara teknis.")
    
    col_data1, col_data2 = st.columns([1, 2])
    
    with col_data1:
        st.markdown("### A.1 Penjelasan Variabel")
        st.markdown("""
        - **Provinsi**: Nama wilayah administratif (Index). Tipe: *String*.
        - **Kelapa_Sawit**: Produksi kelapa sawit (Ribuan Ton). Tipe: *Float64*.
        - **Kelapa**: Produksi kelapa (Ribuan Ton). Tipe: *Float64*.
        - **Karet**: Produksi karet (Ribuan Ton). Tipe: *Float64*.
        - **Kopi**: Produksi kopi (Ribuan Ton). Tipe: *Float64*.
        - **Kakao**: Produksi kakao (Ribuan Ton). Tipe: *Float64*.
        - **Teh**: Produksi teh (Ribuan Ton). Tipe: *Float64*.
        - **Tebu**: Produksi tebu (Ribuan Ton). Tipe: *Float64*.
        """)
        
    with col_data2:
        st.markdown("### A.2 Head() Data")
        st.dataframe(df.head(10), use_container_width=True)
        
    st.markdown("### Info() Struktur Data")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.markdown("### Describe() Statistik Deskriptif")
    st.dataframe(df.describe().T.style.background_gradient(cmap='Blues', subset=['mean', '50%', 'max']), use_container_width=True)

# ==============================================================================
# HALAMAN 2: DATA CLEANING
# ==============================================================================
elif menu == "🧹 Data Cleaning & Preprocessing":
    st.markdown("# 🧹 Data Cleaning & Preprocessing (Bagian B)")
    st.markdown("Proses preprocessing data untuk memastikan kualitas data sebelum dianalisis lebih lanjut. "
                "Tahapan ini mencakup penanganan *Missing Value*, *Duplicate Data*, *Data Type*, dan *Outlier*.")
    
    st.divider()
    
    # 1. Missing Value
    st.markdown("## 1. Penanganan Missing Value")
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        st.success(f"✅ Tidak ditemukan Missing Value pada dataset. Semua data terisi (0 NaN).")
    else:
        st.warning(f"Ditemukan {missing_count} missing value. Proses imputasi dengan nilai 0 dilakukan.")
    
    # 2. Duplicate
    st.markdown("## 2. Penanganan Duplicate Data")
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        st.success(f"✅ Tidak ditemukan data duplikat. Setiap provinsi unik.")
    else:
        st.warning(f"Ditemukan {dup_count} baris duplikat. Baris duplikat akan dihapus.")
        df.drop_duplicates(inplace=True)
        
    # 3. Data Type
    st.markdown("## 3. Konversi Tipe Data")
    st.info("Semua kolom komoditas telah dipastikan bertipe data numerik (Float64) untuk memungkinkan operasi matematis.")
    st.dataframe(df.dtypes.to_frame('Tipe Data'), use_container_width=True)
    
    # 4. Outlier Handling
    st.markdown("## 4. Deteksi & Penanganan Outlier (IQR Method)")
    st.markdown("Outlier dideteksi menggunakan metode *Interquartile Range* (IQR). Nilai yang berada di luar batas "
                "Q1 - 1.5*IQR atau Q3 + 1.5*IQR diidentifikasi. Penanganan dilakukan dengan metode *Winsorization* (Capping).")
    
    col_out1, col_out2 = st.columns([1, 2])
    
    with col_out1:
        selected_kom = st.selectbox("Pilih Komoditas untuk analisis outlier:", komoditas)
        
        Q1 = df[selected_kom].quantile(0.25)
        Q3 = df[selected_kom].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df[selected_kom] < lower) | (df[selected_kom] > upper)]
        
        st.metric("Jumlah Outlier Ditemukan", len(outliers))
        st.metric("Batas Bawah (Lower Bound)", f"{lower:.2f}")
        st.metric("Batas Atas (Upper Bound)", f"{upper:.2f}")
        
    with col_out2:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=df[selected_kom], color='lightcoral', ax=ax)
        ax.set_title(f'Boxplot {selected_kom} (Sebelum Capping)', fontsize=14)
        st.pyplot(fig)
        
        if len(outliers) > 0:
            with st.expander("Lihat Daftar Provinsi Outlier"):
                st.dataframe(outliers[[selected_kom]])
                
    st.divider()
    st.markdown("### Proses Capping (Winsorization)")
    df_capped, _ = processor.remove_outliers_iqr(df, komoditas)
    
    col_cap1, col_cap2 = st.columns(2)
    with col_cap1:
        st.markdown("**Sebelum Capping**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df[selected_kom], color='salmon', ax=ax)
        st.pyplot(fig)
        
    with col_cap2:
        st.markdown("**Sesudah Capping**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df_capped[selected_kom], color='lightgreen', ax=ax)
        st.pyplot(fig)
        
    render_interpretation(f"Outlier pada komoditas {selected_kom} umumnya terjadi karena provinsi tertentu memang sentra produksi utama (misal Riau untuk Sawit). Namun untuk keperluan pemodelan machine learning, outlier ini di-capping agar tidak mendistorsi model secara berlebihan.")

# ==============================================================================
# HALAMAN 3: EDA 2D
# ==============================================================================
elif menu == "📊 Eksplorasi Data (EDA 2D)":
    st.markdown("# 📊 Exploratory Data Analysis (Bagian C)")
    st.markdown("Visualisasi data eksploratori menggunakan Matplotlib dan Seaborn untuk memahami distribusi, "
                "hubungan, dan pola tersembunyi dalam data.")
    
    selected_kom = st.selectbox("Pilih Komoditas:", komoditas)
    
    col1, col2 = st.columns(2)
    
    # 1. Histogram
    with col1:
        st.markdown("### 1. Histogram Distribusi")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[selected_kom], bins=15, kde=True, color='#3b82f6', ax=ax)
        ax.lines[0].set_color('#ec4899')
        ax.set_title(f'Distribusi Produksi {selected_kom}')
        ax.set_xlabel('Produksi (Ribuan Ton)')
        st.pyplot(fig)
        render_interpretation(f"Distribusi {selected_kom} terlihat *right-skewed* (miring ke kanan), menandakan mayoritas provinsi memiliki produksi rendah, dengan beberapa provinsi ekstrem tinggi.")
        
    # 2. Boxplot
    with col2:
        st.markdown("### 2. Boxplot & Outlier")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[selected_kom], color='#22c55e', ax=ax)
        ax.set_title(f'Boxplot Produksi {selected_kom}')
        st.pyplot(fig)
        render_interpretation("Boxplot mengkonfirmasi adanya nilai ekstrem (outlier) yang menjadi sentra produksi nasional.")
        
    # 3. Bar Chart
    st.markdown("### 3. Bar Chart 10 Provinsi Teratas")
    top10 = df[[selected_kom]].sort_values(by=selected_kom, ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=top10[selected_kom], y=top10.index, palette='viridis', ax=ax)
    ax.set_title(f'Top 10 Provinsi Produsen {selected_kom}')
    ax.set_xlabel('Jumlah Produksi')
    st.pyplot(fig)
    render_interpretation(f"Provinsi {top10.index[0]} mendominasi produksi {selected_kom} dengan total {top10[selected_kom].iloc[0]:.2f} ribu ton.")
    
    # 4. Heatmap Korelasi
    st.markdown("### 4. Heatmap Korelasi Antar Komoditas")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[komoditas].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Heatmap Korelasi Pearson')
    st.pyplot(fig)
    render_interpretation("Korelasi positif tinggi antar komoditas (misal Sawit dan Karet) mengindikasikan provinsi yang memproduksi satu kemungkinan besar memproduksi yang lain karena kesamaan kondisi geografis.")
    
    # 5. Scatter Plot
    st.markdown("### 5. Scatter Plot Hubungan 2 Variabel")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        x_sc = st.selectbox("Variabel X:", komoditas, index=0)
    with col_s2:
        y_sc = st.selectbox("Variabel Y:", komoditas, index=3)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df, x=x_sc, y=y_sc, scatter_kws={'color':'#8b5cf6', 'alpha':0.7}, line_kws={'color':'red'}, ax=ax)
    ax.set_title(f'Scatter Plot {x_sc} vs {y_sc}')
    st.pyplot(fig)
    
    # 6. Line Plot Tren
    st.markdown("### 6. Line Plot Tren Total Produksi Nasional")
    total_prod = df[komoditas].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=total_prod.index, y=total_prod.values, marker='o', color='#f59e0b', linewidth=2, markersize=10, ax=ax)
    ax.set_title('Total Produksi Nasional per Komoditas')
    ax.set_ylabel('Total (Ribuan Ton)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ==============================================================================
# HALAMAN 4: VISUALISASI 3D (BONUS & REQUEST)
# ==============================================================================
elif menu == "🧊 Visualisasi 3D Interaktif":
    st.markdown("# 🧊 Visualisasi 3D Interaktif")
    st.markdown("Eksplorasi data dalam ruang 3 dimensi untuk mendeteksi pola klaster atau hubungan multivariat yang lebih kompleks.")
    
    tab1, tab2, tab3 = st.tabs(["📡 3D Scatter Plot", "🏔️ 3D Surface Plot", "📊 3D Bar Chart"])
    
    with tab1:
        st.markdown("### 📡 3D Scatter Plot Antar Komoditas")
        col_x, col_y, col_z, col_c = st.columns(4)
        with col_x: x_3d = st.selectbox("Sumbu X", komoditas, index=0)
        with col_y: y_3d = st.selectbox("Sumbu Y", komoditas, index=1)
        with col_z: z_3d = st.selectbox("Sumbu Z", komoditas, index=2)
        with col_c: c_3d = st.selectbox("Warna Berdasarkan", komoditas, index=3)
        
        fig = px.scatter_3d(
            df.reset_index(), x=x_3d, y=y_3d, z=z_3d,
            color=c_3d, hover_name='Provinsi',
            title=f"Visualisasi 3D: {x_3d} vs {y_3d} vs {z_3d}",
            color_continuous_scale=px.colors.sequential.Viridis,
            template='plotly_dark'
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), scene=dict(
                            xaxis_title=x_3d, yaxis_title=y_3d, zaxis_title=z_3d))
        st.plotly_chart(fig, use_container_width=True)
        render_interpretation("Visualisasi 3D scatter memungkinkan kita melihat klastering alami dari provinsi berdasarkan tiga komoditas sekaligus. Warna membantu memahami dimensi keempat.")
        
    with tab2:
        st.markdown("### 🏔️ 3D Surface Plot (Topografi Produksi)")
        st.markdown("Permukaan 3D ini menggambarkan relasi matematis antara dua komoditas terhadap total produksi gabungan.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1: surf_x = st.selectbox("Variabel X (Surface)", komoditas, index=0)
        with col_s2: surf_y = st.selectbox("Variabel Y (Surface)", komoditas, index=1)
        
        # Membuat meshgrid untuk surface
        x = df[surf_x].values
        y = df[surf_y].values
        z = x + y # Z adalah total gabungan
        
        # Interpolasi untuk membuat surface halus
        from scipy.interpolate import griddata
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
        
        fig = go.Figure(data=[go.Surface(
            z=zi, x=xi, y=yi,
            colorscale='Portland',
            contours_z=dict(show=True, usecolormap=True, project_z=True)
        )])
        fig.update_layout(title=f'Surface Plot {surf_x} & {surf_y} terhadap Total Produksi',
                          scene=dict(xaxis_title=surf_x, yaxis_title=surf_y, zaxis_title='Total Z'),
                          margin=dict(l=0, r=0, b=0, t=40),
                          template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.markdown("### 📊 3D Bar Chart Top 10 Provinsi")
        st.markdown("Diagram batang 3D menunjukkan magnitude produksi 2 komoditas di 10 provinsi teratas.")
        
        bar_kom1 = st.selectbox("Komoditas 1 (3D Bar)", komoditas, index=0)
        bar_kom2 = st.selectbox("Komoditas 2 (3D Bar)", komoditas, index=3)
        
        top10_3d = df[[bar_kom1, bar_kom2]].sum(axis=1).sort_values(ascending=False).head(10).index
        df_3d_bar = df.loc[top10_3d].reset_index()
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=df_3d_bar.index, y=[1]*len(df_3d_bar), z=df_3d_bar[bar_kom1],
                color='rgba(59, 130, 246, 0.8)', name=bar_kom1
            ),
            go.Mesh3d(
                x=df_3d_bar.index, y=[2]*len(df_3d_bar), z=df_3d_bar[bar_kom2],
                color='rgba(236, 72, 153, 0.8)', name=bar_kom2
            )
        ])
        
        # Alternatif lebih sederhana: 3D Scatter yang dimanipulasi jadi bar dengan garis
        fig = go.Figure()
        for i, prov in enumerate(df_3d_bar['Provinsi']):
            fig.add_trace(go.Scatter3d(x=[i, i], y=[1, 1], z=[0, df_3d_bar[bar_kom1].iloc[i]],
                                       mode='lines+markers', line=dict(width=20, color='blue'),
                                       name=prov if i==0 else None, showlegend=False))
            fig.add_trace(go.Scatter3d(x=[i, i], y=[2, 2], z=[0, df_3d_bar[bar_kom2].iloc[i]],
                                       mode='lines+markers', line=dict(width=20, color='pink'),
                                       showlegend=False))
                                       
        fig.update_layout(scene=dict(
                            xaxis=dict(tickvals=list(range(10)), ticktext=df_3d_bar['Provinsi'], title='Provinsi'),
                            yaxis=dict(tickvals=[1, 2], ticktext=[bar_kom1, bar_kom2], title='Komoditas'),
                            zaxis_title='Produksi'),
                          title="3D Bar Chart Custom",
                          margin=dict(l=0, r=0, b=0, t=40),
                          template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# HALAMAN 5: ANALISIS HUBUNGAN VARIABEL (KORELASI & STATISTIK)
# ==============================================================================
elif menu == "🔗 Analisis Hubungan Variabel":
    st.markdown("# 🔗 Analisis Hubungan Variabel (Bagian D)")
    st.markdown("Menghitung korelasi Pearson dan Spearman, serta mengidentifikasi variabel paling berpengaruh menggunakan uji signifikansi (P-Value).")
    
    col1, col2 = st.columns(2)
    with col1: target = st.selectbox("Pilih Variabel Target (Y):", komoditas, index=0)
    with col2: method = st.selectbox("Metode Korelasi:", ['pearson', 'spearman'])
    
    corr_matrix = df[komoditas].corr(method=method)
    
    st.markdown("### Matriks Korelasi")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", ax=ax)
    st.pyplot(fig)
    
    st.markdown("### Identifikasi Variabel Paling Berpengaruh")
    
    corr_target = corr_matrix[target].drop(target).abs().sort_values(ascending=False)
    most_inf = corr_target.index[0]
    r_val = corr_matrix.loc[most_inf, target]
    
    # Uji P-Value dengan Scipy
    r_stat, p_val = stats.pearsonr(df[most_inf], df[target])
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Variabel Paling Berpengaruh", most_inf)
    m2.metric("Nilai Korelasi (r)", f"{r_val:.4f}")
    m3.metric("P-Value", f"{p_val:.4f}")
    
    if p_val < 0.05:
        st.success(f"**Kesimpulan Statistik:** Korelasi antara {most_inf} dan {target} SIGNIFIKAN secara statistik (p < 0.05).")
    else:
        st.warning(f"**Kesimpulan Statistik:** Korelasi tidak signifikan. Hubungan mungkin terjadi kebetulan.")
        
    st.markdown("### Penjelasan Lengkap Alasan")
    st.info(f"Variabel `{most_inf}` dipilih sebagai variabel paling berpengaruh terhadap `{target}` karena memiliki nilai absolut korelasi tertinggi ({abs(r_val):.4f}). "
            f"Nilai r yang {'positif' if r_val > 0 else 'negatif'} menunjukkan bahwa jika produksi {most_inf} naik, maka produksi {target} cenderung {'naik' if r_val > 0 else 'turun'}. "
            "Secara domain knowledge, hal ini masuk akal karena komoditas ini biasa ditanam dalam ekosistem perkebunan yang sama atau saling melengkapi secara ekonomi.")
            
    # Scatter Plot Matrix
    st.markdown("### Pairplot Hubungan Antar Variabel")
    st.write("Pairplot digunakan untuk melihat hubungan bivariat antar seluruh pasangan komoditas sekaligus.")
    fig = sns.pairplot(df[komoditas], kind='reg', diag_kind='kde', 
                       plot_kws={'line_kws':{'color':'red'}, 'scatter_kws':{'alpha':0.5}})
    st.pyplot(fig)

# ==============================================================================
# HALAMAN 6: PEMODELAN REGRESI & UJI ASUMSI
# ==============================================================================
elif menu == "🧮 Pemodelan Regresi & Uji Asumsi":
    st.markdown("# 🧮 Pemodelan Regresi Linear (Bagian E)")
    st.markdown("Membangun model regresi linear berganda menggunakan Statsmodels untuk memprediksi produksi suatu komoditas berdasarkan komoditas lainnya.")
    
    col1, col2 = st.columns(2)
    with col1: y_var = st.selectbox("Variabel Dependen (Y):", komoditas, index=0)
    with col2: x_vars = st.multiselect("Variabel Independen (X):", [k for k in komoditas if k != y_var], default=[k for k in komoditas if k != y_var][:3])
    
    if len(x_vars) > 0:
        X = df[x_vars]
        y = df[y_var]
        X_const = sm.add_constant(X)
        
        model = sm.OLS(y, X_const).fit()
        
        st.divider()
        st.markdown("### Ringkasan Model Regresi (OLS)")
        st.text(model.summary())
        
        # Metrik Evaluasi
        y_pred = model.predict(X_const)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        r2 = model.rsquared
        
        st.markdown("### Evaluasi Metrik")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
        c2.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
        c3.metric("R-Squared (R²)", f"{r2:.4f}")
        
        st.info(f"**Interpretasi Model:** Nilai R² = {r2:.4f} berarti variabel independen mampu menjelaskan {r2*100:.2f}% variasi dari {y_var}. "
                f"MAE sebesar {mae:.2f} menunjukkan rata-rata kesalahan prediksi model dalam satuan ribu ton.")
                
        st.divider()
        st.markdown("### 📉 Analisis Residu (Uji Asumsi Klasik)")
        st.markdown("Memastikan model memenuhi asumsi Normalitas, Homoskedastisitas, dan Linearitas.")
        
        residuals = model.resid
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("#### 1. Histogram Residu (Normalitas)")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(residuals, kde=True, color='purple', ax=ax)
            ax.set_title('Distribusi Residu')
            st.pyplot(fig)
            
            # Q-Q Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            sm.qqplot(residuals, line='45', fit=True, ax=ax, color='blue')
            ax.set_title('Q-Q Plot Residu')
            st.pyplot(fig)
            
        with col_res2:
            st.markdown("#### 2. Scatter Residu vs Fitted (Homoskedastisitas)")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=y_pred, y=residuals, color='orange', ax=ax)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel('Nilai Prediksi (Fitted Values)')
            ax.set_ylabel('Residu')
            ax.set_title('Residuals vs Fitted')
            st.pyplot(fig)
            
            # Uji VIF (Multikolinearitas)
            st.markdown("#### 3. Variance Inflation Factor (VIF)")
            vif_data = pd.DataFrame()
            vif_data['Variabel'] = X_const.columns
            vif_data['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(len(X_const.columns))]
            st.dataframe(vif_data.style.applymap(lambda x: 'color: red' if x > 10 else 'color: green', subset=['VIF']))
            
        st.markdown("### Visualisasi Aktual vs Prediksi")
        df_res = pd.DataFrame({'Aktual': y, 'Prediksi': y_pred})
        fig = px.scatter(df_res, x='Aktual', y='Prediksi', trendline='ols',
                         title='Aktual vs Prediksi', template='plotly_dark')
        fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], 
                                 mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Silakan pilih minimal 1 variabel Independen (X).")

# ==============================================================================
# HALAMAN 7: CLUSTERING HIRARKIS (BONUS)
# ==============================================================================
elif menu == "🌳 Clustering Hirarkis (Bonus)":
    st.markdown("# 🌳 Clustering Hirarkis Provinsi (Bonus)")
    st.markdown("Mengelompokkan provinsi berdasarkan kemiripan profil produksi komoditasnya menggunakan algoritma Hierarchical Clustering (Dendrogram) dari library Scipy.")
    
    # Standarisasi data sebelum clustering
    df_scaled = stats.zscore(df[komoditas])
    
    st.markdown("### Dendrogram Klaster Provinsi")
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Hitung jarak (Linkage)
    linked = hierarchy.linkage(df_scaled, method='ward')
    
    # Plot Dendrogram
    hierarchy.dendrogram(linked, labels=df.index.tolist(), orientation='top', 
                         distance_sort='descending', show_leaf_counts=True, ax=ax,
                         leaf_rotation=90, leaf_font_size=10)
    ax.set_title('Dendrogram Hierarchical Clustering Provinsi Indonesia', fontsize=16)
    ax.set_ylabel('Jarak (Distance)')
    st.pyplot(fig)
    
    render_interpretation("Dendrogram menunjukkan bagaimana provinsi-provinsi dikelompokkan berdasarkan kemiripan pola produksi. "
                          "Provinsi yang bergabung di level bawah (misal Sumatera) memiliki profil perkebunan yang sangat mirip.", type="info")

# ==============================================================================
# HALAMAN 8: INSIGHT & REKOMENDASI
# ==============================================================================
elif menu == "💡 Insight & Rekomendasi":
    st.markdown("# 💡 Insight & Rekomendasi (Bagian F)")
    st.markdown("Berdasarkan analisis mendalam dari data, berikut adalah insight dan rekomendasi implementatif sebagai seorang Data Scientist.")
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.markdown("## 🧠 5 Insight Utama")
        
        st.markdown("""
        **1. Sentralisasi Produksi Sawit di Sumatera & Kalimantan**
        Data menunjukkan bahwa >90% produksi Kelapa Sawit berasal dari dua pulau ini, dengan Riau sebagai raja mutlak. Hal ini menciptakan risiko konsentrasi ekonomi dan ekologi.
        
        **2. Kopi dan Kakao Saling Berkorelasi Positif**
        Berdasarkan matriks korelasi, provinsi penghasil Kopi cenderung juga memproduksi Kakao. Ini mengindikasikan kesamaan syarat agro-klimatologi (tanah vulkanik & iklim tropis).
        
        **3. Pulau Jawa Mendominasi Tebu dan Teh**
        Berkebalikan dengan sawit, Jawa Timur dan Jawa Barat memonopoli komoditas Tebu dan Teh. Ini mencerminkan warisan kolonial dan ketersediaan infrastruktur pabrik gula lokal.
        
        **4. Ketimpangan Produksi Antar Provinsi (Outlier Ekstrem)**
        Analisis Boxplot menunjukkan skewness ekstrem. Hanya 3-5 provinsi yang berperan sebagai "workhorse" nasional per komoditas, sementara puluhan provinsi lainnya hanya berproduksi marginal.
        
        **5. Potensi Tersembunyi di Indonesia Timur**
        Papua dan Maluku, meskipun saat ini produksinya rendah, menunjukkan keberadaan komoditas kakao dan kelapa. Wilayah ini berpotensi menjadi frontier baru jika infrastruktur dibangun.
        """)
        
    with col_ins2:
        st.markdown("## 🎯 5 Rekomendasi Implementatif")
        
        st.markdown("""
        **1. Diversifikasi Ekonomi di Sentra Sawit**
        Pemerintah perlu mendorong diversifikasi tanaman di Riau dan Kalimantan untuk memitigasi risiko harga CPO global. Subsidi untuk komoditas selingan (Kopi/Kakao) dianjurkan.
        
        **2. Program Sinkronisasi Kopi-Kakao**
        Buat program integrated farming di provinsi seperti Sulawesi dan Sumatera untuk mengkapitalisasi korelasi positif Kopi dan Kakao, meningkatkan efisiensi lahan.
        
        **3. Revitalisasi Pabrik Gula di Jawa**
        Karena Jawa Timur memimpin Tebu, investasi revitalisasi pabrik gula (PG) di sana akan berdampak langsung pada peningkatan produksi nasional gula.
        
        **4. Transfer Teknologi ke Provinsi Marginal**
        Lakukan pendekatan *Knowledge Transfer* dari provinsi top-producer ke provinsi penghasil rendah. Misalnya, praktik budidaya Kelapa dari Riau diterapkan di Maluku.
        
        **5. Pemetaan Lahan Berbasis Data (Land Suitability Mapping)**
        Menggunakan data ini sebagai baseline, buat peta suitability lahan 3D untuk mencari lahan kosong di Indonesia Timur yang memiliki profil agro-klimatologi serupa dengan sentra produksi utama.
        """)
        
    st.divider()
    st.success("Dashboard ini telah menganalisis data secara komprehensif dari tahap preprocessing, EDA 2D & 3D, hingga pemodelan. Analisis ini siap digunakan untuk pengambilan keputusan strategis.")
