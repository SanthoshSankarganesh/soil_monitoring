import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import base64
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
# Set page config
st.set_page_config(page_title="Soil Health Analyzer", layout="wide")

# Inject responsive style
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 16px;
        }
        @media screen and (max-width: 768px) {
            html, body, [class*="css"] {
                font-size: 14px;
            }
        }
    </style>
""", unsafe_allow_html=True)

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

# Helper
@st.cache_data
def load_and_prep_image(image_data):
    image = Image.open(image_data).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    return img_array

# Soil information
def get_soil_data():
    return {
        "Sand": {
            "Recommended Crops": "Sand supports root vegetables like carrots, radishes, and potatoes due to its loose structure. It’s also good for melons and peanuts.",
            "Nutrient Deficiency": "Sand often lacks nitrogen, potassium, and magnesium, making fertilization essential.",
            "Recommended Fertilizers": "Use slow-release fertilizers rich in nitrogen and potassium. Organic compost and green manure are helpful too.",
            "Tips to Improve Soil": "Incorporate organic matter, such as compost or well-rotted manure, to improve water retention and nutrient content.",
            "Map": [27.0, 72.6]
        },
        "Silt": {
            "Recommended Crops": "Silt is ideal for crops like wheat, rice, and vegetables such as broccoli and lettuce due to good water retention.",
            "Nutrient Deficiency": "May be deficient in nitrogen and phosphorus over time.",
            "Recommended Fertilizers": "Use balanced NPK fertilizers. Adding organic compost can improve structure and fertility.",
            "Tips to Improve Soil": "Prevent compaction by avoiding over-tilling and walking on wet soil. Mix in organic material.",
            "Map": [25.5, 85.1]
        },
        "Clay": {
            "Recommended Crops": "Clay supports crops like rice, broccoli, and cabbage, which tolerate wet soil and benefit from rich nutrients.",
            "Nutrient Deficiency": "May lack calcium and can have pH issues, limiting nutrient availability.",
            "Recommended Fertilizers": "Apply lime to manage pH. Use gypsum and compost to improve texture and nutrient availability.",
            "Tips to Improve Soil": "Add organic matter and sand to reduce compaction and improve drainage. Mulching helps as well.",
            "Map": [23.2, 78.6]
        },
        "Loam": {
            "Recommended Crops": "Loam is excellent for almost all crops including vegetables, fruits, and cereals due to its balanced properties.",
            "Nutrient Deficiency": "Generally fertile but can be low in micronutrients like zinc.",
            "Recommended Fertilizers": "Use organic compost to maintain fertility. Supplement with micronutrients if required.",
            "Tips to Improve Soil": "Practice crop rotation and add organic mulch to retain fertility and structure.",
            "Map": [22.9, 77.5]
        },
        "Peat": {
            "Recommended Crops": "Good for root vegetables and berries. Often used in horticulture due to high organic content.",
            "Nutrient Deficiency": "Lacks minerals such as iron, manganese, and copper.",
            "Recommended Fertilizers": "Use mineral fertilizers rich in micronutrients. Lime may be needed to raise pH.",
            "Tips to Improve Soil": "Improve drainage with sand and manage acidity with lime. Avoid over-watering.",
            "Map": [26.7, 88.4]
        },
        "Chalk": {
            "Recommended Crops": "Suitable for barley, beets, cabbage, and spinach which can tolerate alkaline conditions.",
            "Nutrient Deficiency": "Commonly lacks iron, manganese, and zinc due to high pH.",
            "Recommended Fertilizers": "Use chelated micronutrient sprays. Apply compost and acidic organic matter.",
            "Tips to Improve Soil": "Add gypsum and organic mulch to increase retention and balance alkalinity.",
            "Map": [29.4, 76.9]
        },
        "Alluvial Soil": {
            "Recommended Crops": "Perfect for crops like rice, wheat, sugarcane, and pulses due to fertility and moisture retention.",
            "Nutrient Deficiency": "Can be low in nitrogen and phosphorus in some regions.",
            "Recommended Fertilizers": "Use NPK fertilizers and organic compost to maintain fertility.",
            "Tips to Improve Soil": "Practice sustainable irrigation and use green manure. Maintain crop diversity.",
            "Map": [25.6, 83.0]
        },
        "Black Cotton Soil (Regur)": {
            "Recommended Crops": "Ideal for cotton, soybean, and pulses. Supports oilseeds like sunflower.",
            "Nutrient Deficiency": "Often lacks phosphorus and nitrogen.",
            "Recommended Fertilizers": "Apply phosphorus-rich fertilizers. Organic manure and compost are also helpful.",
            "Tips to Improve Soil": "Deep plowing and gypsum application help reduce cracking. Crop rotation improves productivity.",
            "Map": [19.1, 75.3]
        },
        "Red and Yellow Soil": {
            "Recommended Crops": "Supports groundnuts, millets, potatoes, and coarse grains.",
            "Nutrient Deficiency": "Low in nitrogen, phosphorus, and organic content.",
            "Recommended Fertilizers": "Use NPK fertilizers, farmyard manure, and green manure.",
            "Tips to Improve Soil": "Add compost and lime to increase pH and organic matter.",
            "Map": [20.4, 85.8]
        },
        "Laterite Soil": {
            "Recommended Crops": "Good for cashew, tea, coffee, and rubber due to drainage and acidity.",
            "Nutrient Deficiency": "Lacks nitrogen, phosphorus, and potassium.",
            "Recommended Fertilizers": "Use heavy doses of organic matter and nitrogen-rich fertilizers.",
            "Tips to Improve Soil": "Incorporate mulch and organic compost to boost fertility. Control erosion with cover crops.",
            "Map": [14.4, 74.0]
        }
    }

soil_data = get_soil_data()

# Sidebar navigation
st.sidebar.title("🌱 Soil Analyzer Navigation")
page = st.sidebar.radio("Select Section", [
    "Upload & Predict", "Recommended Crops", "Nutrient Deficiency",
    "Recommended Fertilizers", "Tips to Improve Soil",
    "Soil Distribution Map"])

# Upload & Predict Page
if page == "Upload & Predict":
    st.title("📷 Upload Soil Image & Predict Type")
    uploaded_file = st.file_uploader("Upload a soil image", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False, label_visibility="visible")
    camera_input = st.camera_input("Or take a live photo")

    active_input = camera_input if camera_input else uploaded_file

    if active_input:
        image_data = load_and_prep_image(active_input)
        prediction = model.predict(image_data)[0]
        predicted_index = np.argmax(prediction)
        predicted_soil = soil_labels[predicted_index]
        confidence = prediction[predicted_index]

        st.session_state.image = active_input
        st.session_state.prediction = predicted_soil
        st.session_state.confidence = confidence

        st.success(f"✅ Predicted Soil Type: {predicted_soil} ({confidence*100:.2f}%)")

        fig, ax = plt.subplots()
        ax.bar(soil_labels[:-1], prediction[:-1])
        ax.set_xticklabels(soil_labels[:-1], rotation=45, ha='right')
        ax.set_ylabel("Confidence")
        st.pyplot(fig)

        coords = soil_data.get(predicted_soil, {}).get("Map")
        if coords:
            m = folium.Map(location=coords, zoom_start=5)
            folium.Marker(coords, tooltip=predicted_soil).add_to(m)
            st_folium(m, width=700, height=500)
    else:
        st.warning("📌 Please upload or capture a soil image to continue.")

# Subpage rendering
if page in ["Recommended Crops", "Nutrient Deficiency", "Recommended Fertilizers", "Tips to Improve Soil"]:
    st.title(f"📖 {page}")
    soil = st.session_state.prediction
    if soil and soil in soil_data:
        st.markdown(f"<div style='font-size:18px;'>{soil_data[soil][page]}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Upload and predict a soil image first.")

# Soil Distribution Map
if page == "Soil Distribution Map":
    st.title("🗽 Soil Distribution Map")
    map_center = [22.5, 80.0]
    m = folium.Map(location=map_center, zoom_start=5)
    for soil_type, data in soil_data.items():
        folium.Marker(data["Map"], tooltip=soil_type).add_to(m)
    st_folium(m, width=700, height=500)
