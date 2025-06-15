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
def long_paragraph(text):
    return f"<div style='font-size:18px;'>{text}</div>"

data_template = {
    'Nutrient Deficiency': "Detailed data not available yet.",
    'Recommended Fertilizers': "Detailed data not available yet.",
    'Recommended Crops': "Detailed data not available yet.",
    'Tips to Improve Soil': "Detailed data not available yet.",
    'Map': [20.0, 78.0]
}

soil_data = {
    'Sand': {
        'Nutrient Deficiency': "Sandy soils often lack essential nutrients due to their loose structure, resulting in poor retention of nitrogen, potassium, and phosphorus.",
        'Recommended Fertilizers': "Use slow-release nitrogen fertilizers, organic compost, and biofertilizers to gradually enrich the soil.",
        'Recommended Crops': "Root vegetables like carrots and potatoes, as well as crops like watermelon and peanuts, grow well in sandy soil.",
        'Tips to Improve Soil': "Add compost, mulch heavily, and grow cover crops to build organic matter and improve water retention.",
        'Map': [23.5, 85.2]
    },
    'Silt': {
        'Nutrient Deficiency': "Silts may experience nitrogen depletion and phosphorus lock-up due to their moderate drainage and mineral content.",
        'Recommended Fertilizers': "Apply balanced NPK fertilizers, compost, and micronutrient supplements.",
        'Recommended Crops': "Crops like rice, wheat, and vegetables such as lettuce and broccoli thrive in silt soil.",
        'Tips to Improve Soil': "Incorporate green manure, maintain good drainage, and avoid over-tilling to retain fertility.",
        'Map': [25.5, 83.5]
    },
    'Clay': {
        'Nutrient Deficiency': "Clay soils tend to become compacted, limiting nutrient availability, especially nitrogen and calcium.",
        'Recommended Fertilizers': "Use gypsum, calcium ammonium nitrate, and composted manure to loosen structure and add nutrients.",
        'Recommended Crops': "Soybeans, broccoli, and cabbage grow well in clay with proper aeration.",
        'Tips to Improve Soil': "Add organic matter, avoid working when wet, and install drainage systems to improve texture.",
        'Map': [22.3, 78.4]
    },
    'Loam': {
        'Nutrient Deficiency': "Loamy soils are typically well-balanced but may suffer from micronutrient depletion in intensive farming.",
        'Recommended Fertilizers': "Use compost tea, rock phosphate, and organic fertilizers for sustainable enrichment.",
        'Recommended Crops': "Most crops including maize, wheat, pulses, and vegetables grow excellently in loam.",
        'Tips to Improve Soil': "Rotate crops, mulch regularly, and use compost to sustain fertility.",
        'Map': [26.1, 81.2]
    },
    'Peat': {
        'Nutrient Deficiency': "Peaty soils are acidic and often deficient in minerals like iron, copper, and manganese.",
        'Recommended Fertilizers': "Apply lime to reduce acidity and use chelated micronutrient formulations.",
        'Recommended Crops': "Cranberries, blueberries, and root crops like carrots can tolerate peat soil.",
        'Tips to Improve Soil': "Drain excess water, lime periodically, and add mineral-rich compost.",
        'Map': [28.6, 76.1]
    },
    'Chalk': {
        'Nutrient Deficiency': "Chalky soils often suffer from iron and zinc deficiencies due to high pH.",
        'Recommended Fertilizers': "Use acidic fertilizers, iron sulphate, and organic compost.",
        'Recommended Crops': "Barley, beet, and cabbage do relatively well in chalk with pH management.",
        'Tips to Improve Soil': "Add organic matter to retain moisture and adjust pH using sulfur-based treatments.",
        'Map': [29.3, 75.8]
    },
    'Alluvial Soil': {
        'Nutrient Deficiency': "Alluvial soils may lack nitrogen in intensely farmed areas but are otherwise rich.",
        'Recommended Fertilizers': "Use green manure and supplement with nitrogen-rich fertilizers.",
        'Recommended Crops': "Rice, wheat, sugarcane, and jute grow abundantly in alluvial soil.",
        'Tips to Improve Soil': "Adopt crop rotation, conserve moisture, and apply balanced fertilizers.",
        'Map': [27.2, 83.1]
    },
    'Black Cotton Soil (Regur)': {
        'Nutrient Deficiency': "Typically deficient in nitrogen, phosphorus, and organic matter.",
        'Recommended Fertilizers': "Apply NPK blends with organic manure, and zinc sulphate for micronutrients.",
        'Recommended Crops': "Cotton, sorghum, sunflower, and groundnut are best suited.",
        'Tips to Improve Soil': "Enhance organic matter, practice deep tillage, and prevent water stagnation.",
        'Map': [19.9, 77.4]
    },
    'Red and Yellow Soil': {
        'Nutrient Deficiency': "Generally low in nitrogen, phosphorus, and humus.",
        'Recommended Fertilizers': "Use phosphatic fertilizers and farmyard manure regularly.",
        'Recommended Crops': "Millets, pulses, and oilseeds grow best.",
        'Tips to Improve Soil': "Apply organic compost, avoid erosion, and grow cover crops.",
        'Map': [20.3, 84.1]
    },
    'Laterite Soil': {
        'Nutrient Deficiency': "Highly leached soil, often deficient in nitrogen, potassium, and calcium.",
        'Recommended Fertilizers': "Add dolomite, lime, and balanced fertilizers.",
        'Recommended Crops': "Cashew, tea, and tapioca perform well.",
        'Tips to Improve Soil': "Add compost, practice mulching, and control erosion.",
        'Map': [15.5, 75.6]
    }
}

# Sidebar navigation
st.sidebar.title("🌱 Soil Analyzer Navigation")
page = st.sidebar.radio("Select Section", [
    "Upload & Predict", "Recommended Crops", "Nutrient Deficiency",
    "Recommended Fertilizers", "Tips to Improve Soil",
    "Soil Distribution Map", "Export PDF"])

# Upload & Predict Page
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
            ax.bar(range(len(prediction[:-1])), prediction[:-1])
            ax.set_xticks(range(len(soil_labels[:-1])))
            ax.set_xticklabels(soil_labels[:-1], rotation=45)
            ax.set_ylabel("Confidence")
            ax.set_title("Soil Prediction Probabilities")
            st.pyplot(fig)

# Detail Pages
elif page == "Recommended Crops" and st.session_state.prediction:
    soil = st.session_state.prediction
    st.title(f"🌾 Recommended Crops for {soil}")
    st.markdown(long_paragraph(soil_data[soil]['Recommended Crops']), unsafe_allow_html=True)

elif page == "Nutrient Deficiency" and st.session_state.prediction:
    soil = st.session_state.prediction
    st.title(f"🥀 Nutrient Deficiency in {soil}")
    st.markdown(long_paragraph(soil_data[soil]['Nutrient Deficiency']), unsafe_allow_html=True)

elif page == "Recommended Fertilizers" and st.session_state.prediction:
    soil = st.session_state.prediction
    st.title(f"🌿 Fertilizers Recommended for {soil}")
    st.markdown(long_paragraph(soil_data[soil]['Recommended Fertilizers']), unsafe_allow_html=True)

elif page == "Tips to Improve Soil" and st.session_state.prediction:
    soil = st.session_state.prediction
    st.title(f"🛠️ Tips to Improve {soil}")
    st.markdown(long_paragraph(soil_data[soil]['Tips to Improve Soil']), unsafe_allow_html=True)

elif page == "Soil Distribution Map" and st.session_state.prediction:
    soil = st.session_state.prediction
    lat, lon = soil_data[soil]['Map']
    st.title("🗺️ Soil Type Distribution in India")
    fmap = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker([lat, lon], popup=soil).add_to(fmap)
    st_data = st_folium(fmap, width=700)

# Export PDF Page
elif page == "Export PDF":
    st.title("📄 Export Soil Health Report")
    if st.session_state.prediction:
        soil = st.session_state.prediction
        info = soil_data.get(soil, data_template)

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
        st.warning("Upload and predict a soil image before generating the PDF.")

# Fallback message
else:
    st.warning("Upload and predict a valid soil image first to view details.")
