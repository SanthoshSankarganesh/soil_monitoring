if not os.path.exists("temp"):
    os.makedirs("temp")
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
        .pdf-button-hidden .stButton > button {
            display: none;
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
    "Soil Distribution Map", "Export PDF"])

# Upload & Predict Page
if page == "Upload & Predict":
    st.title("📷 Upload Soil Image & Predict Type")
    uploaded_file = st.file_uploader("Upload a soil image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = load_and_prep_image(uploaded_file)
        prediction = model.predict(image_data)[0]
        predicted_index = np.argmax(prediction)
        predicted_soil = soil_labels[predicted_index]
        confidence = prediction[predicted_index]

        st.session_state.image = uploaded_file
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
        st.warning("📌 Please upload a soil image to continue.")

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
    st.title("🗺 Soil Distribution Map")
    map_center = [22.5, 80.0]
    m = folium.Map(location=map_center, zoom_start=5)
    for soil_type, data in soil_data.items():
        folium.Marker(data["Map"], tooltip=soil_type).add_to(m)
    st_folium(m, width=700, height=500)

# Export PDF
if page == "Export PDF":
    st.title("🧾 Export PDF Report")
    if st.session_state.image and st.session_state.prediction:
        soil = st.session_state.prediction
        content = f"""
        <h2>Soil Health Report</h2>
        <p><strong>Predicted Soil:</strong> {soil}</p>
        <p><strong>Confidence:</strong> {st.session_state.confidence*100:.2f}%</p>
        <p><strong>Recommended Crops:</strong> {soil_data[soil]['Recommended Crops']}</p>
        <p><strong>Nutrient Deficiency:</strong> {soil_data[soil]['Nutrient Deficiency']}</p>
        <p><strong>Recommended Fertilizers:</strong> {soil_data[soil]['Recommended Fertilizers']}</p>
        <p><strong>Tips to Improve Soil:</strong> {soil_data[soil]['Tips to Improve Soil']}</p>
        """
        file_name = f"{soil}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        file_path = os.path.join("temp", file_name)
        os.makedirs("temp", exist_ok=True)
        pdfkit.from_string(content, file_path, configuration=PDFKIT_CONFIG)
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{file_name}">📥 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Please upload and predict a soil image first.")
