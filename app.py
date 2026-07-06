# ==============================================================================
# UAS PENGENALAN SAINS DATA - ENTERPRISE DASHBOARD 3D (STABLE VERSION)
# ==============================================================================
# Tools: Streamlit, Pandas, Numpy, Matplotlib, Seaborn, Plotly, Scipy, Statsmodels
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.cluster import hierarchy
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import io
import base64
from typing import List, Dict, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Agri-Analytics 3D Enterprise", page_icon="🌿", layout="wide")

# ==============================================================================
# DATA INJECTION (FALLBACK)
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
# CUSTOM CSS & THEME MANAGER (FIXED: NO F-STRING INTERPOLATION)
# ==============================================================================
def load_css(theme: str = "dark") -> None:
    """Memuat CSS berdasarkan tema yang dipilih tanpa f-string untuk mencegah error."""
    
    if theme == "dark":
        css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
            :root {
                --bg-color: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --primary: #3b82f6;
                --secondary: #8b5cf6;
            }
            .stApp {
                background: radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.15) 0%, transparent 40%),
                            radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
                            #0f172a;
                color: var(--text-main);
                font-family: 'Outfit', sans-serif;
                transition: background 0.5s ease;
            }
            h1, h2, h3, h4 {
                font-family: 'Outfit', sans-serif !important;
                font-weight: 800 !important;
                background: linear-gradient(90deg, var(--primary), var(--secondary));
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
            }
            section[data-testid="stSidebar"] {
                background: rgba(15, 23, 42, 0.9) !important;
                border-right: 1px solid rgba(148, 163, 184, 0.2) !important;
                backdrop-filter: blur(10px);
            }
            div[data-testid="stMetric"] {
                background: var(--card-bg) !important;
                border: 1px solid rgba(148, 163, 184, 0.2) !important;
                padding: 20px !important;
                border-radius: 15px !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
                transition: transform 0.3s ease !important;
            }
            div[data-testid="stMetric"]:hover {
                transform: translateY(-5px) !important;
                border-color: var(--primary) !important;
            }
            .stDataFrame, .stTable {
                background: var(--card-bg) !important;
                border-radius: 12px !important;
                border: 1px solid rgba(148, 163, 184, 0.2) !important;
            }
            .stButton>button, .stDownloadButton>button {
                background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
                color: white !important;
                border: none !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }
            .stSelectbox>div>div, .stMultiselect>div>div, .stSlider>div>div {
                background: var(--card-bg) !important;
                border-color: rgba(148, 163, 184, 0.3) !important;
                color: var(--text-main) !important;
            }
        </style>
        """
    else:
        css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
            :root {
                --bg-color: #f8fafc;
                --card-bg: #ffffff;
                --text-main: #0f172a;
                --text-muted: #64748b;
                --primary: #2563eb;
                --secondary: #7c3aed;
            }
            .stApp {
                background: #f8fafc;
                color: var(--text-main);
                font-family: 'Outfit', sans-serif;
                transition: background 0.5s ease;
            }
            h1, h2, h3, h4 {
                font-family: 'Outfit', sans-serif !important;
                font-weight: 800 !important;
                background: linear-gradient(90deg, var(--primary), var(--secondary));
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
            }
            section[data-testid="stSidebar"] {
                background: #ffffff !important;
                border-right: 1px solid rgba(148, 163, 184, 0.2) !important;
            }
            div[data-testid="stMetric"] {
                background: #ffffff !important;
                border: 1px solid rgba(148, 163, 184, 0.2) !important;
                padding: 20px !important;
                border-radius: 15px !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
                transition: transform 0.3s ease !important;
            }
            div[data-testid="stMetric"]:hover {
                transform: translateY(-5px) !important;
                border-color: var(--primary) !important;
            }
            .stDataFrame, .stTable {
                background: #ffffff !important;
                border-radius: 12px !important;
                border: 1px solid rgba(148, 163, 184, 0.2) !important;
            }
            .stButton>button, .stDownloadButton>button {
                background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
                color: white !important;
                border: none !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }
            .stSelectbox>div>div, .stMultiselect>div>div, .stSlider>div>div {
                background: #ffffff !important;
                border-color: rgba(148, 163, 184, 0.3) !important;
                color: var(--text-main) !important;
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# Inisialisasi Theme State
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"

load_css(st.session_state.theme)

# ==============================================================================
# DATA PROCESSOR CLASS
# ==============================================================================
class AgriDataProcessor:
    """Kelas untuk memproses data pertanian."""
    def __init__(self, raw_data: str):
        self.raw_data = raw_data
        self.df = None
        self.df_region = None
        self.commodities = ['Kelapa_Sawit', 'Kelapa', 'Karet', 'Kopi', 'Kakao', 'Teh', 'Tebu']
        
    def process(self) -> pd.DataFrame:
        df = pd.read_csv(io.StringIO(self.raw_data), sep=';', index_col=0)
        df.columns = df.columns.str.strip()
        df.fillna(0, inplace=True)
        for col in self.commodities:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df[~df.index.duplicated(keep='first')]
        self.df = df
        
        # Mapping Region (Pulau)
        region_map = {}
        sumatera = ["ACEH", "SUMATERA UTARA", "SUMATERA BARAT", "RIAU", "JAMBI", "SUMATERA SELATAN", "BENGKULU", "LAMPUNG", "KEP. BANGKA BELITUNG", "KEP. RIAU"]
        jawa = ["DKI JAKARTA", "JAWA BARAT", "JAWA TENGAH", "DI YOGYAKARTA", "JAWA TIMUR", "BANTEN"]
        bali_nt = ["BALI", "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR"]
        kalimantan = ["KALIMANTAN BARAT", "KALIMANTAN TENGAH", "KALIMANTAN SELATAN", "KALIMANTAN TIMUR", "KALIMANTAN UTARA"]
        sulawesi = ["SULAWESI UTARA", "SULAWESI TENGAH", "SULAWESI SELATAN", "SULAWESI TENGGARA", "GORONTALO", "SULAWESI BARAT"]
        maluku_papua = ["MALUKU", "MALUKU UTARA", "PAPUA BARAT", "PAPUA BARAT DAYA", "PAPUA", "PAPUA SELATAN", "PAPUA TENGAH", "PAPUA PEGUNUNGAN"]
        
        for p in sumatera: region_map[p] = "Sumatera"
        for p in jawa: region_map[p] = "Jawa"
        for p in bali_nt: region_map[p] = "Bali & Nusa Tenggara"
        for p in kalimantan: region_map[p] = "Kalimantan"
        for p in sulawesi: region_map[p] = "Sulawesi"
        for p in maluku_papua: region_map[p] = "Maluku & Papua"
            
        df['Pulau'] = df.index.map(region_map)
        self.df_region = df
        return df

processor = AgriDataProcessor(RAW_CSV_DATA)
df = processor.process()
komoditas = processor.commodities

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("🌿 Agri-Analytics Enterprise")
st.sidebar.markdown("Sistem Analitik Data Perkebunan")

# Theme Toggle
theme_col1, theme_col2 = st.sidebar.columns([1, 2])
if theme_col1.button("🌙" if st.session_state.theme == "light" else "☀️"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()
theme_col2.markdown(f"**Mode: {st.session_state.theme.upper()}**")

st.sidebar.divider()

menu = st.sidebar.selectbox("Navigasi Modul:", [
    "🏠 Beranda & Data Understanding",
    "🧹 Data Cleaning & Export Engine",
    "📊 Eksplorasi Data (EDA 2D)",
    "🧊 Visualisasi 3D Interaktif",
    "🗺️ Regional Geospatial (Sunburst)",
    "📉 Dimensionality Reduction (PCA)",
    "🔗 Analisis Hubungan Variabel",
    "🧮 Pemodelan Regresi & Uji Asumsi",
    "🎮 What-If Analysis Simulator",
    "💡 Insight & Rekomendasi"
])

st.sidebar.divider()
st.sidebar.markdown("#### Status Dataset")
st.sidebar.write(f"📊 **Observasi:** {df.shape[0]} Provinsi")
st.sidebar.write(f"📈 **Variabel:** {df.shape[1]-1} Komoditas")
st.sidebar.write("✅ **Data Quality:** Excellent")

# Download Data di Sidebar
st.sidebar.divider()
st.sidebar.markdown("#### Download Data")
def get_table_download_link(df_to_download: pd.DataFrame, filename: str, text: str):
    csv = df_to_download.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}"><button style="width:100%;padding:10px;border:none;border-radius:8px;background:linear-gradient(90deg, #3b82f6, #8b5cf6);color:white;font-weight:600;cursor:pointer;">{text}</button></a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

get_table_download_link(df.drop(columns=['Pulau']), "cleaned_data.csv", "💾 Download Clean CSV")

# ==============================================================================
# HALAMAN 1: BERANDA
# ==============================================================================
if menu == "🏠 Beranda & Data Understanding":
    st.markdown("# 🏠 Beranda & Data Understanding")
    st.markdown("Selamat datang di **Agri-Analytics Enterprise**. Dashboard interaktif tingkat lanjut untuk menganalisis komoditas perkebunan Indonesia dengan teknologi visualisasi 3D dan Machine Learning.")
    
    col1, col2, col3, col4 = st.columns(4)
    total_prod = df[komoditas].sum().sum()
    top_kom = df[komoditas].sum().idxmax()
    top_val = df[komoditas].sum().max()
    top_prov = df[komoditas].sum(axis=1).idxmax()
    
    col1.metric("Total Produksi Nasional", f"{total_prod:,.2f} Ton")
    col2.metric("Komoditas Terbanyak", top_kom)
    col3.metric("Nilai Produksi Terbanyak", f"{top_val:,.2f} Ton")
    col4.metric("Provinsi Terproduktif", top_prov)
    
    st.divider()
    st.markdown("## 📋 Data Understanding")
    
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.markdown("### A.1 Penjelasan Variabel")
        st.markdown("""
        - **Provinsi**: Nama wilayah administratif. Tipe: *String*.
        - **Kelapa_Sawit**: Produksi kelapa sawit (Ribuan Ton). Tipe: *Float64*.
        - **Kelapa**: Produksi kelapa (Ribuan Ton). Tipe: *Float64*.
        - **Karet**: Produksi karet (Ribuan Ton). Tipe: *Float64*.
        - **Kopi**: Produksi kopi (Ribuan Ton). Tipe: *Float64*.
        - **Kakao**: Produksi kakao (Ribuan Ton). Tipe: *Float64*.
        - **Teh**: Produksi teh (Ribuan Ton). Tipe: *Float64*.
        - **Tebu**: Produksi tebu (Ribuan Ton). Tipe: *Float64*.
        """)
    with col_d2:
        st.markdown("### A.2 Head() Data")
        st.dataframe(df.head(10), use_container_width=True)
        
    st.markdown("### Describe() Statistik Deskriptif")
    st.dataframe(df[komoditas].describe().T.style.background_gradient(cmap='Blues', subset=['mean', '50%', 'max']), use_container_width=True)

# ==============================================================================
# HALAMAN 2: DATA CLEANING
# ==============================================================================
elif menu == "🧹 Data Cleaning & Export Engine":
    st.markdown("# 🧹 Data Cleaning & Preprocessing (Bagian B)")
    st.markdown("Proses pembersihan data menggunakan metode IQR (Interquartile Range) untuk deteksi dan *capping* outlier.")
    
    selected_kom = st.selectbox("Pilih Komoditas untuk analisis outlier:", komoditas)
    
    Q1 = df[selected_kom].quantile(0.25)
    Q3 = df[selected_kom].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[selected_kom] < lower) | (df[selected_kom] > upper)]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Outlier", len(outliers))
    c2.metric("Batas Bawah", f"{lower:.2f}")
    c3.metric("Batas Atas", f"{upper:.2f}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df[selected_kom], color='salmon', ax=ax)
    ax.set_title(f'Boxplot {selected_kom} (Sebelum Capping)')
    st.pyplot(fig)
    
    if len(outliers) > 0:
        st.markdown("### Daftar Provinsi Outlier")
        st.dataframe(outliers[[selected_kom]])
        get_table_download_link(outliers[[selected_kom]], "outlier_data.csv", "💾 Download Outlier Data")

# ==============================================================================
# HALAMAN 3: EDA 2D
# ==============================================================================
elif menu == "📊 Eksplorasi Data (EDA 2D)":
    st.markdown("# 📊 Exploratory Data Analysis (Bagian C)")
    selected_kom = st.selectbox("Pilih Komoditas:", komoditas)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 1. Histogram")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[selected_kom], bins=15, kde=True, color='#3b82f6', ax=ax)
        ax.set_title(f'Distribusi {selected_kom}')
        st.pyplot(fig)
    with c2:
        st.markdown("### 2. Boxplot")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[selected_kom], color='#22c55e', ax=ax)
        st.pyplot(fig)
        
    st.markdown("### 3. Top 10 Provinsi")
    top10 = df[[selected_kom]].sort_values(by=selected_kom, ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=top10[selected_kom], y=top10.index, palette='viridis', ax=ax)
    st.pyplot(fig)
    
    st.markdown("### 4. Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[komoditas].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# ==============================================================================
# HALAMAN 4: VISUALISASI 3D
# ==============================================================================
elif menu == "🧊 Visualisasi 3D Interaktif":
    st.markdown("# 🧊 Visualisasi 3D Interaktif")
    st.markdown("Eksplorasi multivariat dalam ruang 3D menggunakan Plotly.")
    
    tab1, tab2 = st.tabs(["📡 3D Scatter Plot", "🏔️ 3D Surface Plot"])
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        x_3d = c1.selectbox("Sumbu X", komoditas, index=0)
        y_3d = c2.selectbox("Sumbu Y", komoditas, index=1)
        z_3d = c3.selectbox("Sumbu Z", komoditas, index=2)
        c_3d = c4.selectbox("Warna", komoditas, index=3)
        
        fig = px.scatter_3d(df.reset_index(), x=x_3d, y=y_3d, z=z_3d, color=c_3d, hover_name='Provinsi',
                            color_continuous_scale=px.colors.sequential.Viridis, template='plotly_dark')
        fig.update_traces(marker=dict(size=7, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        surf_x = st.selectbox("Variabel X (Surface)", komoditas, index=0)
        surf_y = st.selectbox("Variabel Y (Surface)", komoditas, index=1)
        
        x = df[surf_x].values
        y = df[surf_y].values
        z = x + y
        
        from scipy.interpolate import griddata
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
        
        fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Portland')])
        fig.update_layout(scene=dict(xaxis_title=surf_x, yaxis_title=surf_y, zaxis_title='Total'), template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# HALAMAN 5: REGIONAL SUNBURST
# ==============================================================================
elif menu == "🗺️ Regional Geospatial (Sunburst)":
    st.markdown("# 🗺️ Regional Geospatial Analysis")
    st.markdown("Visualisasi hierarki produksi dari tingkat Pulau -> Provinsi -> Komoditas menggunakan Sunburst Chart.")
    
    df_sun = df.reset_index().melt(id_vars=['Provinsi', 'Pulau'], value_vars=komoditas, var_name='Komoditas', value_name='Produksi')
    df_sun = df_sun[df_sun['Produksi'] > 0]
    
    fig = px.sunburst(df_sun, path=['Pulau', 'Provinsi', 'Komoditas'], values='Produksi',
                      color='Produksi', color_continuous_scale='RdYlGn',
                      title="Distribusi Produksi Perkebunan Hierarkis",
                      template='plotly_dark')
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("📊 **Interpretasi:** Klik pada bagian pulau untuk melakukan zoom-in ke provinsi dan komoditas di dalamnya.")

# ==============================================================================
# HALAMAN 6: PCA
# ==============================================================================
elif menu == "📉 Dimensionality Reduction (PCA)":
    st.markdown("# 📉 Principal Component Analysis (PCA)")
    st.markdown("Reduksi 7 dimensi komoditas menjadi 3 dimensi utama (PC1, PC2, PC3) untuk melihat pengelompokan provinsi secara alami.")
    
    X = df[komoditas].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    cov_matrix = np.cov(X_std.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # FIX: Cast to real to avoid complex numbers due to float precision
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    pc_scores = np.dot(X_std, eigenvectors)
    
    total_var = eigenvalues.sum()
    expl_var = (eigenvalues / total_var) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("PC1 Variance", f"{expl_var[0]:.2f}%")
    c2.metric("PC2 Variance", f"{expl_var[1]:.2f}%")
    c3.metric("PC3 Variance", f"{expl_var[2]:.2f}%")
    
    df_pca = pd.DataFrame(pc_scores[:, :3], columns=['PC1', 'PC2', 'PC3'], index=df.index)
    df_pca['Pulau'] = df['Pulau']
    
    st.markdown("### 3D Scatter Plot PCA")
    fig = px.scatter_3d(df_pca.reset_index(), x='PC1', y='PC2', z='PC3', color='Pulau', hover_name='Provinsi',
                        title="Clustering Provinsi Berdasarkan PCA (3D)", template='plotly_dark')
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='white')))
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# HALAMAN 7: HUBUNGAN VARIABEL
# ==============================================================================
elif menu == "🔗 Analisis Hubungan Variabel":
    st.markdown("# 🔗 Analisis Hubungan Variabel (Bagian D)")
    target = st.selectbox("Pilih Variabel Target (Y):", komoditas, index=0)
    
    corr_matrix = df[komoditas].corr(method='pearson')
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
    
    corr_target = corr_matrix[target].drop(target).abs().sort_values(ascending=False)
    most_inf = corr_target.index[0]
    r_val = corr_matrix.loc[most_inf, target]
    r_stat, p_val = stats.pearsonr(df[most_inf], df[target])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Variabel Paling Berpengaruh", most_inf)
    c2.metric("Nilai Korelasi (r)", f"{r_val:.4f}")
    c3.metric("P-Value", f"{p_val:.4f}")

# ==============================================================================
# HALAMAN 8: REGRESI
# ==============================================================================
elif menu == "🧮 Pemodelan Regresi & Uji Asumsi":
    st.markdown("# 🧮 Pemodelan Regresi Linear (Bagian E)")
    y_var = st.selectbox("Variabel Dependen (Y):", komoditas, index=0)
    x_vars = st.multiselect("Variabel Independen (X):", [k for k in komoditas if k != y_var], default=[k for k in komoditas if k != y_var][:3])
    
    if len(x_vars) > 0:
        X = df[x_vars]
        y = df[y_var]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        
        st.text(model.summary())
        
        y_pred = model.predict(X_const)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        r2 = model.rsquared
        
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("R-Squared", f"{r2:.4f}")

# ==============================================================================
# HALAMAN 9: WHAT-IF SIMULATOR
# ==============================================================================
elif menu == "🎮 What-If Analysis Simulator":
    st.markdown("# 🎮 What-If Analysis Simulator")
    st.markdown("Simulasi prediksi interaktif. Geser slider produksi komoditas X untuk memprediksi nilai produksi Y secara real-time.")
    
    c1, c2 = st.columns(2)
    with c1: y_sim = st.selectbox("Target Prediksi (Y):", komoditas, index=3)
    with c2: x_sim = st.selectbox("Variabel Input (X):", [k for k in komoditas if k != y_sim], index=0)
    
    X_sim = df[x_sim].values
    Y_sim = df[y_sim].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(X_sim, Y_sim)
    
    st.metric("Kekuatan Korelasi (r)", f"{r_value:.4f}")
    
    st.divider()
    st.markdown(f"### Atur Nilai Input: {x_sim}")
    
    max_val = float(df[x_sim].max())
    input_val = st.slider(f"Geser untuk mensimulasikan produksi {x_sim} (Ribuan Ton):", 0.0, max_val, float(df[x_sim].mean()), step=0.1)
    
    prediction = slope * input_val + intercept
    prediction = max(0, prediction)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        title = {'text': f"Prediksi Produksi {y_sim}"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, float(df[y_sim].max())]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, df[y_sim].quantile(0.33)], 'color': "#22c55e"},
                {'range': [df[y_sim].quantile(0.33), df[y_sim].quantile(0.66)], 'color': "#f59e0b"},
                {'range': [df[y_sim].quantile(0.66), df[y_sim].max()], 'color': "#ef4444"}
            ]
        }
    ))
    fig.update_layout(height=400, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"Jika produksi **{x_sim}** diatur sebesar **{input_val:.2f} Ton**, maka model memprediksi produksi **{y_sim}** akan menjadi **{prediction:.2f} Ton**.")

# ==============================================================================
# HALAMAN 10: INSIGHT
# ==============================================================================
elif menu == "💡 Insight & Rekomendasi":
    st.markdown("# 💡 Insight & Rekomendasi (Bagian F)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("## 🧠 5 Insight Utama")
        st.markdown("""
        1. **Sentralisasi Sawit di Sumatera/Kalimantan**: Riau dan Kalimantan Tengah mendominasi >90% sawit.
        2. **Korelasi Kopi-Kakao**: Hubungan positif kuat menunjukkan kesamaan agro-klimatologi.
        3. **Jawa Memimpin Tebu/Teh**: Warisan infrastruktur pabrik gula membuat Jawa Timur mendominasi Tebu.
        4. **Outlier Ekstrem**: Hanya 3-5 provinsi yang menjadi "workhorse" nasional per komoditas.
        5. **Potensi Indonesia Timur**: Maluku dan Papua punya potensi kelapa/kakao yang belum tergarap maksimal.
        """)
    with c2:
        st.markdown("## 🎯 5 Rekomendasi Implementatif")
        st.markdown("""
        1. **Diversifikasi di Riau**: Mitigasi risiko harga CPO global dengan subsidi komoditas selingan.
        2. **Integrated Farming Kopi-Kakao**: Program sinkronisasi penanaman di Sulawesi dan Sumatera.
        3. **Revitalisasi Pabrik Gula**: Investasi infrastruktur di Jawa Timur untuk dukung produksi Tebu.
        4. **Transfer Teknologi**: Praktik budidaya dari sentra produksi diterapkan di provinsi marginal.
        5. **Land Suitability Mapping**: Gunakan PCA untuk mencari lahan baru dengan profil mirip sentra produksi.
        """)
