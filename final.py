import hydralit as hy
import streamlit as st
import numpy as np
from datetime import datetime
import joblib
import requests
import xml.etree.ElementTree as ET
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import folium
import base64

# Initialize session state
st.session_state.sync = True
st.session_state.allow_access = True

# Memuat model regresi linear yang telah disimpan
loaded_regressor = joblib.load('model_RandomForest.pkl')
loaded_rf_regressor = joblib.load('model_regresi_linear.pkl')

# Load Earthquakes.csv into a pandas DataFrame
gempa_dataset = pd.read_csv('Earthquakes.csv')

app = hy.HydraApp(title='Earthquake Prediction App')

@app.addapp(title='HOME', icon="🏠")
def home():
    hy.image('back.png', use_column_width=True)

@app.addapp(title='REGRESSION PREDICTION', icon="📈")
def regression_prediction():
    col1_width = 100
    col2_width = 800
    col3_width = 100
    dol1, dol2, dol3 = hy.columns([col1_width, col2_width, col3_width])
    with dol1:
        hy.markdown("")
    with dol2:
        hy.markdown("<h2 style='text-align: center;'>Prediction App</h2>", unsafe_allow_html=True)
        input_date = hy.date_input('Select date', key='linear_date_input', min_value=datetime(1900, 1, 1))
        if input_date:
            tanggal = input_date.toordinal()
            latitude = hy.number_input('Latitude', key='linear_latitude_input', value=0.0)
            longitude = hy.number_input('Longitude', key='linear_longitude_input', value=0.0)
            depth = hy.number_input('Depth (km)', key='linear_depth_input', value=0)

        if hy.button("Predict", key='predict_button'):
            hy.markdown("")
            
    col1_width = 450
    col2_width = 100
    col3_width = 450
    col1, col2, col3 = hy.columns([col1_width, col2_width, col3_width])
    with col1:
        contoh_data = np.array([[tanggal, latitude, longitude, depth]])
        prediksi_magnitude = loaded_regressor.predict(contoh_data)[0]
        
        hy.markdown("#### Linear Regression Prediction")
        hy.write(f"Predicted Magnitude: {prediksi_magnitude:.2f}")
        
        if prediksi_magnitude <= 1.5:
            fol1, fol2 = hy.columns(2)
            with fol1:
                hy.image('1.png', use_column_width=True)
            with fol2:
                hy.success("Earthquake Classification: Small Earthquake")
        elif 1.5 < prediksi_magnitude <= 3.0:
            fol3, fol4 = hy.columns(2)
            with fol3:
                hy.image('2.png', use_column_width=True)
            with fol4:
                hy.warning("Earthquake Classification: Moderate Earthquake")
        else:
            fol5,fol6 = hy.columns(2)
            with fol5:
                hy.image('3.png', use_column_width=True)
            with fol6:
                hy.error("Earthquake Classification: Large Earthquake")
    with col2:
        hy.markdown("")
    with col3:
        contoh_data_rf = np.array([[tanggal, latitude, longitude, depth]])
        prediksi_magnitude_rf = loaded_rf_regressor.predict(contoh_data_rf)[0]
        
        hy.markdown("#### Random Forest Regression Prediction")
        hy.write(f"Predicted Magnitude: {prediksi_magnitude_rf:.2f}")
        
        if prediksi_magnitude_rf <= 1.5:
            mol1, mol2 = hy.columns(2)
            with mol1:
                hy.image('1.png', use_column_width=True)
            with mol2:
                hy.success("Earthquake Classification: Small Earthquake")
        elif 1.5 < prediksi_magnitude_rf <= 3.0:
            mol3, mol4 = hy.columns(2)
            with mol3:
                hy.image('2.png', use_column_width=True)
            with mol4:
                hy.warning("Earthquake Classification: Moderate Earthquake")
        else:
            mol5, mol6 = hy.columns(2)
            with mol5:
                hy.image('3.png', use_column_width=True)
            with mol6:
                hy.error("Earthquake Classification: Large Earthquake")
    with dol3:
        hy.markdown("")    

@app.addapp(title='EARTHQUAKE UPDATE', icon="📌")
def latest_earthquakes():
    hy.markdown("<h2 style='text-align: center;'>Update Earthquake Information (M ≥ 5.0)</h2>", unsafe_allow_html=True)
    hy.markdown("")
    hy.markdown("")
    hy.markdown("")
    hy.markdown("")

    # Fetch data from BMKG
    url = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.xml"
    response = requests.get(url)

    if response.status_code == 200:
        # Parse XML data
        data = ET.fromstring(response.content)

        data_list = []
        i = 1
        for gempaM5 in data.findall('.//gempa'):
            data_list.append({
                "No": i,
                "Tanggal": gempaM5.find('Tanggal').text,
                "Jam": gempaM5.find('Jam').text,
                "DateTime": gempaM5.find('DateTime').text,
                "Magnitudo": gempaM5.find('Magnitude').text,
                "Kedalaman": gempaM5.find('Kedalaman').text,
                "Koordinat": gempaM5.find('point/coordinates').text,
                "Lintang": gempaM5.find('Lintang').text,
                "Bujur": gempaM5.find('Bujur').text,
                "Lokasi": gempaM5.find('Wilayah').text,
                "Potensi": gempaM5.find('Potensi').text
            })
            i += 1

        if hy.checkbox('Show the latest earthquake data'):
            hy.subheader('Raw data')
            hy.table(data_list)
    else:
        hy.write("Failed to fetch data from BMKG.")
        
    hy.markdown("")
    hy.markdown("")
    hy.markdown("")
    col1, col2 = hy.columns(2)
    with col1:
        # Histogram atau Bar Chart Magnitudo:
        hy.subheader('Histogram of the Earthquake Magnitude')
        magnitudo_values = [float(gempa["Magnitudo"]) for gempa in data_list]
        plt.figure(figsize=(10, 6))
        plt.hist(magnitudo_values, bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('Magnitudo')
        plt.ylabel('Jumlah Kejadian')
        plt.grid(axis='y', alpha=0.75)
        hy.pyplot(plt)
        
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
        
        # Membuat peta dengan marker gempa
        hy.subheader('Map with Markers')
        m = folium.Map(location=[-2, 120], zoom_start=5)
        for gempa in data_list:
            magnitudo = float(gempa["Magnitudo"])
            koordinat = [float(coord) for coord in gempa["Koordinat"].split(',')]
            lokasi = gempa["Lokasi"]
            folium.Marker(location=koordinat, popup=f"Magnitudo: {magnitudo}\nLokasi: {lokasi}").add_to(m)
        m.save('gempa_map.html')

        # Use st.components.html to embed the HTML file
        iframe_html = '<iframe src="data:text/html;base64,' + base64.b64encode(open('gempa_map.html', 'r').read().encode()).decode() + '" width=800 height=600></iframe>'
        st.components.v1.html(iframe_html, width=750, height=600)
        
    with col2:
        # Scatter Plot Kedalaman vs. Magnitudo:
        hy.subheader('Scatter Plot Depth vs. Magnitudo')
        kedalaman_values = [float(gempa["Kedalaman"].split()[0]) for gempa in data_list]
        magnitudo_values = [float(gempa["Magnitudo"]) for gempa in data_list]

        plt.figure(figsize=(10, 6))
        plt.scatter(kedalaman_values, magnitudo_values, alpha=0.5)
        plt.xlabel('Kedalaman (km)')
        plt.ylabel('Magnitudo')
        plt.grid(True)
        hy.pyplot(plt)
        
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
        
        # Time Series Plot:
        hy.subheader('Time Series Plot - Number of Earthquakes Over Time')
        tanggal_values = []

        for gempa in data_list:
            date_str = gempa["DateTime"]
            
            # Parsing the date string to handle the provided format
            parsed_date = datetime.strptime(date_str.split("T")[0], "%Y-%m-%d")
            tanggal_values.append(parsed_date)

        jumlah_gempa = range(1, len(tanggal_values) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(tanggal_values, jumlah_gempa, marker='o')
        plt.xlabel('Tanggal')
        plt.ylabel('Jumlah Gempa')
        plt.grid(True)
        hy.pyplot(plt)


        
        
@app.addapp(title='ABOUT', icon="⚙")
def about():
    hy.title("About the Creator")
    col1_width = 200
    col2_width = 800
    col1, col2 = hy.columns([col1_width, col2_width])
    with col1:
        hy.image('girl.png', use_column_width=True)
    with col2:
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
        hy.markdown("")
        hy.markdown("""
            **Creator:**
            Dita Nuraini Adhiharta

            **Program Studi:**
            Teknologi Informasi

            **NIM:**
            21537141003

            **State University of Yogyakarta**
        """)

# Run the whole lot, we get navbar, state management, and app isolation, all with this tiny amount of work.
app.run()
