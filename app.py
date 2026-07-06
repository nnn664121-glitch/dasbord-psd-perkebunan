import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Produksi Tanaman", page_icon="🌱", layout="wide")

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    # Membaca data dengan separator titik koma dan index pertama
    try:
        df = pd.read_csv('produksi_tanaman.csv', sep=';', index_col=0)
    except:
        # Fallback jika ternyata dipisah koma
        df = pd.read_csv('produksi_tanaman.csv', index_col=0)
    
    # Membersihkan nama kolom dari spasi
    df.columns = df.columns.str.strip()
    # Membersihkan missing value (jika ada) dengan 0
    df.fillna(0, inplace=True)
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("🌱 Dashboard Pertanian")
st.sidebar.markdown("UAS Pengenalan Sains Data")
st.sidebar.markdown("**Analisis Komoditas Perkebunan**")
st.sidebar.divider()

menu = st.sidebar.radio("Pilih Menu Analisis:", [
    "📊 Data Understanding", 
    "📈 Exploratory Data Analysis (EDA)", 
    "🔗 Analisis Hubungan Variabel", 
    "🧮 Pemodelan Regresi"
])

# List komoditas
komoditas = ['Kelapa_Sawit', 'Kelapa', 'Karet', 'Kopi', 'Kakao', 'Teh', 'Tebu']

# --- KONTEN UTAMA ---

if menu == "📊 Data Understanding":
    st.title("Data Understanding")
    st.write("Dataset produksi tanaman perkebunan per provinsi di Indonesia.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Observasi (Provinsi)", df.shape[0])
    with col2:
        st.metric("Jumlah Variabel", df.shape[1])
    with col3:
        st.metric("Total Komoditas Dianalisis", len(komoditas))
    
    st.divider()
    
    st.subheader("1. Tampilkan Head() Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("2. Info() Tipe Data")
    # Membuat dataframe info manual untuk ditampilkan di streamlit
    df_info = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.count().values,
        'Dtype': df.dtypes.values
    })
    st.dataframe(df_info, use_container_width=True)
    
    st.subheader("3. Describe() Statistik Deskriptif")
    st.dataframe(df.describe().T, use_container_width=True)


elif menu == "📈 Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Visualisasi data menggunakan Matplotlib dan Seaborn.")
    
    selected_komoditas = st.selectbox("Pilih Komoditas untuk Visualisasi:", komoditas)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"1. Histogram Distribusi {selected_komoditas}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[selected_komoditas], bins=15, kde=True, color='skyblue', ax=ax)
        ax.set_xlabel(f'Produksi {selected_komoditas}')
        ax.set_ylabel('Frekuensi')
        st.pyplot(fig)
        st.caption("Interpretasi: Melihat sebaran data produksi. Apakah data terdistribusi normal atau skewed (miring).")
        
    with col2:
        st.subheader(f"2. Boxplot {selected_komoditas}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[selected_komoditas], color='lightgreen', ax=ax)
        ax.set_xlabel(f'Produksi {selected_komoditas}')
        st.pyplot(fig)
        st.caption("Interpretasi: Mendeteksi adanya outlier (pencilan) pada data provinsi tertentu yang produksinya jauh di atas rata-rata.")

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader(f"3. Bar Chart 10 Provinsi Teratas ({selected_komoditas})")
        top10 = df[[selected_komoditas]].sort_values(by=selected_komoditas, ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top10[selected_komoditas], y=top10.index, palette='viridis', ax=ax)
        ax.set_xlabel('Jumlah Produksi')
        ax.set_ylabel('Provinsi')
        st.pyplot(fig)
        st.caption("Interpretasi: Menunjukkan provinsi mana yang menjadi sentral produksi komoditas tersebut.")

    with col4:
        st.subheader("4. Heatmap Korelasi Antar Komoditas")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_matrix = df[komoditas].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.caption("Interpretasi: Melihat kekuatan hubungan antar komoditas. Nilai mendekati 1 berarti positif sangat kuat.")
        
    st.divider()
    st.subheader("5. Scatter Plot & Line Plot")
    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        x_scatter = st.selectbox("Pilih Variabel X (Scatter):", komoditas, index=0)
        y_scatter = st.selectbox("Pilih Variabel Y (Scatter):", komoditas, index=3)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=x_scatter, y=y_scatter, hue='Provinsi', legend=False, ax=ax)
        ax.set_xlabel(x_scatter)
        ax.set_ylabel(y_scatter)
        st.pyplot(fig)
    
    with sc_col2:
        # Line plot total produksi per komoditas
        total_produksi = df[komoditas].sum().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x=total_produksi.index, y=total_produksi.values, marker='o', color='purple', ax=ax)
        ax.set_xlabel('Komoditas')
        ax.set_ylabel('Total Produksi')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.caption("Interpretasi: Melihat tren total produksi secara keseluruhan di Indonesia.")


elif menu == "🔗 Analisis Hubungan Variabel":
    st.title("Analisis Hubungan Variabel (Korelasi)")
    
    st.write("Menghitung korelasi Pearson dan mengidentifikasi variabel paling berpengaruh menggunakan **SciPy**.")
    
    # Hitung korelasi
    corr_matrix = df[komoditas].corr()
    
    st.subheader("Matriks Korelasi")
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
    
    st.divider()
    st.subheader("Identifikasi Variabel Paling Berpengaruh")
    
    target_var = st.selectbox("Pilih Variabel Target (Y):", komoditas, index=0)
    
    # Menghitung korelasi terhadap variabel target
    corr_target = corr_matrix[target_var].drop(target_var).abs().sort_values(ascending=False)
    most_influential = corr_target.index[0]
    corr_value = corr_target.iloc[0]
    
    # Uji signifikansi (p-value) dengan Scipy
    r, p_value = stats.pearsonr(df[most_influential], df[target_var])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Variabel Paling Berpengaruh", most_influential)
        st.metric("Nilai Korelasi (r)", f"{corr_value:.4f}")
    
    with col2:
        st.metric("P-Value", f"{p_value:.4f}")
        if p_value < 0.05:
            st.success("Signifikan secara statistik (p < 0.05)")
        else:
            st.warning("Tidak signifikan secara statistik (p >= 0.05)")
            
    st.write(f"**Penjelasan:** Variabel `{most_influential}` memiliki korelasi tertinggi terhadap `{target_var}` dengan nilai r = {corr_value:.4f}. Alasannya, secara matematis nilai kovarians dan deviasi standar antara kedua variabel ini menunjukkan hubungan linear yang paling kuat dibandingkan pasangan variabel lainnya.")


elif menu == "🧮 Pemodelan Regresi":
    st.title("Pemodelan Regresi Linear")
    st.write("Membangun model regresi menggunakan **Statsmodels**.")
    
    col1, col2 = st.columns(2)
    with col1:
        y_var = st.selectbox("Pilih Variabel Dependen (Y):", komoditas, index=0)
    with col2:
        x_vars = st.multiselect("Pilih Variabel Independen (X):", [k for k in komoditas if k != y_var], default=[k for k in komoditas if k != y_var][:2])
    
    if len(x_vars) > 0:
        # Siapkan data
        X = df[x_vars]
        y = df[y_var]
        
        # Tambahkan konstanta untuk intercept
        X = sm.add_constant(X)
        
        # Buat model OLS (Ordinary Least Squares)
        model = sm.OLS(y, X).fit()
        
        st.divider()
        st.subheader("Ringkasan Model Regresi")
        # Tampilkan summary model
        st.text(model.summary())
        
        # Ekstrak metrik
        y_pred = model.predict(X)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        r2 = model.rsquared
        
        st.divider()
        st.subheader("Evaluasi Metrik Model")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
        m2.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
        m3.metric("R² (R-Squared)", f"{r2:.4f}")
        
        st.write("**Interpretasi Hasil Model:**")
        st.write(f"- **R² = {r2:.4f}**: Artinya, sekitar {r2*100:.2f}% variabilitas dari `{y_var}` dapat dijelaskan oleh variabel independen yang dipilih ({', '.join(x_vars)}).")
        st.write(f"- **MAE = {mae:.2f}**: Rata-rata kesalahan absolut prediksi sebesar {mae:.2f} unit produksi.")
        st.write(f"- **RMSE = {rmse:.2f}**: Akar kuadrat dari rata-rata kesalahan kuadrat. Memberikan penalti lebih besar untuk error yang besar.")
        
        st.divider()
        st.subheader("Visualisasi Prediksi vs Aktual (Plotly)")
        # Membuat dataframe untuk plotly
        df_plot = pd.DataFrame({'Aktual': y, 'Prediksi': y_pred})
        
        fig = px.scatter(df_plot, x='Aktual', y='Prediksi', 
                         title='Perbandingan Nilai Aktual vs Prediksi',
                         labels={'Aktual': 'Nilai Aktual', 'Prediksi': 'Nilai Prediksi'},
                         trendline='ols',
                         template='plotly_white')
        # Garis ideal y=x
        fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], 
                                 mode='lines', name='Garis Ideal (Y=X)', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Silakan pilih minimal satu variabel Independen (X).")
