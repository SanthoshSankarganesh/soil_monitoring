import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pdfkit
from datetime import datetime
import base64
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# Configure wkhtmltopdf path
PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")

# Set page config
st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

# Initialize session state
if 'image' not in st.session_state:
    st.session_state.image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

# Soil types
soil_labels = [
    'Sand', 'Silt', 'Clay', 'Loam', 'Peat', 'Chalk',
    'Alluvial Soil', 'Black Cotton Soil (Regur)', 'Red and Yellow Soil', 'Laterite Soil', 'Unknown'
]

# Load model
model = tf.keras.models.load_model('keras_model.h5', compile=False)

# Soil data
data_template = {
    'Nutrient Deficiency': "No data available.",
    'Recommended Fertilizers': "No data available.",
    'Recommended Crops': "No data available.",
    'Tips to Improve Soil': "No data available.",
    'Map': [20.0, 78.0]
}

soil_data = {
    'Sand': {
        'Nutrient Deficiency': "Sandy soils are often low in essential nutrients like nitrogen and phosphorus due to high leaching.",
        'Recommended Fertilizers': "Use organic compost, green manure, and slow-release nitrogen fertilizers.",
        'Recommended Crops': "Carrots, radishes, peanuts, and watermelon thrive in sandy soil.",
        'Tips to Improve Soil': "Incorporate organic matter, add biochar, and mulch frequently to retain moisture.",
        'Map': [23.5, 85.2]
    },
    'Silt': {
        'Nutrient Deficiency': "Silts may suffer from poor phosphorus availability and compaction-related issues.",
        'Recommended Fertilizers': "Balanced NPK fertilizers and organic compost are beneficial.",
        'Recommended Crops': "Wheat, soybean, sugarcane, and vegetables.",
        'Tips to Improve Soil': "Enhance structure by adding organic matter and avoid over-tilling.",
        'Map': [26.0, 87.5]
    },
    'Clay': {
        'Nutrient Deficiency': "Clay soil may suffer from poor aeration and potassium deficiency.",
        'Recommended Fertilizers': "Use potassium-rich fertilizers and gypsum to improve soil structure.",
        'Recommended Crops': "Rice, broccoli, and lettuce are suitable for clay soil.",
        'Tips to Improve Soil': "Add compost and gypsum to reduce compaction and improve drainage.",
        'Map': [25.0, 81.0]
    },
    'Loam': {
        'Nutrient Deficiency': "Generally fertile, but may sometimes lack nitrogen if overused.",
        'Recommended Fertilizers': "Use balanced organic or synthetic fertilizers.",
        'Recommended Crops': "Almost all crops including maize, wheat, pulses, and vegetables.",
        'Tips to Improve Soil': "Rotate crops, maintain pH, and use cover crops.",
        'Map': [22.0, 78.0]
    },
    'Peat': {
        'Nutrient Deficiency': "Deficient in micronutrients like copper, iron, and zinc.",
        'Recommended Fertilizers': "Micronutrient sprays and well-rotted compost are ideal.",
        'Recommended Crops': "Root crops, salad greens, and brassicas.",
        'Tips to Improve Soil': "Mix with mineral soil, lime to reduce acidity, and improve drainage.",
        'Map': [26.7, 88.4]
    },
    'Chalk': {
        'Nutrient Deficiency': "Often lacks iron and manganese due to high alkalinity.",
        'Recommended Fertilizers': "Use chelated iron, organic matter, and acidic fertilizers.",
        'Recommended Crops': "Barley, clover, spinach, and beet.",
        'Tips to Improve Soil': "Add organic matter and acidic mulches to balance pH.",
        'Map': [30.3, 76.4]
    },
    'Alluvial Soil': {
        'Nutrient Deficiency': "Sometimes low in nitrogen and organic carbon.",
        'Recommended Fertilizers': "Green manure, compost, and urea-based nitrogen fertilizers.",
        'Recommended Crops': "Rice, wheat, sugarcane, and pulses.",
        'Tips to Improve Soil': "Regular use of organic amendments and crop rotation.",
        'Map': [25.6, 84.9]
    },
    'Black Cotton Soil (Regur)': {
        'Nutrient Deficiency': "Often deficient in nitrogen, phosphorus, and organic carbon.",
        'Recommended Fertilizers': "Use NPK mixtures and farmyard manure.",
        'Recommended Crops': "Cotton, soybeans, sorghum, and pulses.",
        'Tips to Improve Soil': "Apply organic compost and manage irrigation carefully.",
        'Map': [19.1, 77.4]
    },
    'Red and Yellow Soil': {
        'Nutrient Deficiency': "Typically low in nitrogen, phosphorus, and humus.",
        'Recommended Fertilizers': "Phosphatic and nitrogen-rich fertilizers with compost.",
        'Recommended Crops': "Millets, pulses, groundnut, and oilseeds.",
        'Tips to Improve Soil': "Use organic compost, green manure, and mulching.",
        'Map': [20.8, 83.2]
    },
    'Laterite Soil': {
        'Nutrient Deficiency': "Deficient in nitrogen, phosphorus, and potassium.",
        'Recommended Fertilizers': "Heavy use of compost, cow dung manure, and lime.",
        'Recommended Crops': "Tea, coffee, cashew, and tapioca.",
        'Tips to Improve Soil': "Add lime to reduce acidity and regular organic manure.",
        'Map': [15.3, 75.1]
    }
}

# Sidebar navigation
st.sidebar.title("🌱 Soil Analyzer Navigation")
page = st.sidebar.radio("Select Section", [
    "Upload & Predict", "Recommended Crops", "Nutrient Deficiency",
    "Recommended Fertilizers", "Tips to Improve Soil",
    "Soil Distribution Map", "Export PDF"])

# Prediction and visualization logic
if page == "Upload & Predict":
    st.title("📷 Upload Soil Image")
    uploaded_file = st.file_uploader("Upload a soil image", type=["jpg", "jpeg", "png"])
    camera_input = st.camera_input("Or capture using camera")

    if uploaded_file:
        image = Image.open(uploaded_file).resize((224, 224))
        st.session_state.image = image
    elif camera_input:
        image = Image.open(camera_input).resize((224, 224))
        st.session_state.image = image

    if st.session_state.image:
        st.image(st.session_state.image, caption="Uploaded Image", use_column_width=True)
        img_array = np.array(st.session_state.image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        pred_index = int(np.argmax(prediction))
        confidence = float(prediction[pred_index])
        predicted_soil = soil_labels[pred_index]

        if confidence < 0.60 or predicted_soil == "Unknown":
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.error("❌ Not a valid soil image. Upload a clear close-up photo of real soil.")
        else:
            st.session_state.prediction = predicted_soil
            st.session_state.confidence = confidence
            st.success(f"✅ Detected: {predicted_soil} (Confidence: {confidence:.2f})")

            # Chart
            fig, ax = plt.subplots()
            ax.bar(soil_labels[:len(prediction)], prediction)
            ax.set_ylabel("Confidence")
            ax.set_title("Soil Prediction Probabilities")
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Subpages with detailed info
elif st.session_state.prediction:
    soil = st.session_state.prediction
    info = soil_data.get(soil, data_template)

    if page == "Recommended Crops":
        st.title(f"🌾 Recommended Crops for {soil}")
        st.markdown(f"<div style='font-size:18px;'>{info['Recommended Crops']}</div>", unsafe_allow_html=True)

    elif page == "Nutrient Deficiency":
        st.title(f"🥀 Nutrient Deficiency in {soil}")
        st.markdown(f"<div style='font-size:18px;'>{info['Nutrient Deficiency']}</div>", unsafe_allow_html=True)

    elif page == "Recommended Fertilizers":
        st.title(f"🌿 Fertilizers Recommended for {soil}")
        st.markdown(f"<div style='font-size:18px;'>{info['Recommended Fertilizers']}</div>", unsafe_allow_html=True)

    elif page == "Tips to Improve Soil":
        st.title(f"🛠️ Tips to Improve {soil}")
        st.markdown(f"<div style='font-size:18px;'>{info['Tips to Improve Soil']}</div>", unsafe_allow_html=True)

    elif page == "Soil Distribution Map":
        st.title("🗺️ Soil Type Distribution in India")
        lat, lon = info['Map']
        fmap = folium.Map(location=[lat, lon], zoom_start=6)
        folium.Marker([lat, lon], popup=soil).add_to(fmap)
        st_data = st_folium(fmap, width=700)

    elif page == "Export PDF":
        st.title("📄 Export Soil Health Report")
        if st.button("Generate PDF Report"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{soil.replace(' ', '_')}_{timestamp}.pdf"

            pdf_content = f"""
            <h1>Soil Health Report</h1>
            <h2>Soil Type: {soil}</h2>
            <h3>Confidence: {st.session_state.confidence:.2f}</h3>
            <p><b>Nutrient Deficiency:</b> {info['Nutrient Deficiency']}</p>
            <p><b>Recommended Fertilizers:</b> {info['Recommended Fertilizers']}</p>
            <p><b>Recommended Crops:</b> {info['Recommended Crops']}</p>
            <p><b>Tips to Improve Soil:</b> {info['Tips to Improve Soil']}</p>
            """
            pdfkit.from_string(pdf_content, file_name, configuration=PDFKIT_CONFIG)
            with open(file_name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">📥 Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            os.remove(file_name)
else:
    st.warning("Upload and predict a valid soil image first to view details.")

