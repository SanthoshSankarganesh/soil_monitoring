# soilAI.py

import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import pdfkit
from datetime import datetime
from streamlit_folium import st_folium
import folium
import base64
import io
import matplotlib.pyplot as plt

# Configure PDFKit to use wkhtmltopdf path
pdfkit_config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")

# Set page config
st.set_page_config(page_title="Soil Health Monitoring", layout="wide")

# Load the trained model
MODEL_PATH = "keras_model.h5"  # Update with correct path to your model file
model = tf.keras.models.load_model(MODEL_PATH)

# Define all soil types
soil_types = [
    "Sand", "Silt", "Clay", "Loam", "Peat", "Chalk",
    "Alluvial Soil", "Black Cotton Soil (Regur)", "Red and Yellow Soil", "Laterite Soil"
]

# Detailed soil_info with large paragraphs for each topic and soil type
soil_info = {
    "Sand": {
        "Crops Cultivatable": "Sandy soil is ideal for root crops like carrots, radishes, and potatoes, which require well-draining, loose textures. It also supports crops such as peanuts, watermelon, and cucumbers due to its ability to warm up quickly and provide good aeration. However, it requires more frequent irrigation and nutrient management.",
        "Nutrient Deficiency": "Sandy soil often lacks essential nutrients like nitrogen, potassium, and phosphorus due to its high drainage capacity. Organic matter content is typically very low, making it poor at retaining water and nutrients for long durations.",
        "Recommended Fertilizers": "Apply compost and well-rotted manure regularly to build organic matter. Use slow-release nitrogen fertilizers or liquid feeds to address nutrient leaching. Mulching helps reduce evaporation.",
        "Tips to Improve Soil": "Mix organic materials such as compost, peat moss, and vermiculite to improve moisture retention and nutrient-holding capacity. Use ground covers and mulch to reduce wind and water erosion. Implement crop rotation and cover cropping."
    },
    "Silt": {
        "Crops Cultivatable": "Silt soils, being fertile and holding moisture well, support crops like rice, wheat, and vegetables. Their smooth texture aids root expansion, while high fertility ensures consistent yields. They are often found near rivers.",
        "Nutrient Deficiency": "May suffer from compaction and poor drainage over time, leading to lower oxygen availability for roots. Nutrient levels can drop with overuse of chemical fertilizers without organic matter replacement.",
        "Recommended Fertilizers": "Add well-rotted manure or green manure to enhance structure and fertility. Phosphorus and potassium supplements improve root growth and flowering in vegetable crops.",
        "Tips to Improve Soil": "To avoid crusting, reduce heavy foot or machinery traffic. Add organic material and sand to improve aeration. Regular tilling and using raised beds help drainage."
    },
    "Clay": {
        "Crops Cultivatable": "Clay soils are nutrient-rich and retain moisture well, ideal for crops like broccoli, cabbage, and leafy greens. Their density also supports rice cultivation in paddy fields where water stagnation is beneficial.",
        "Nutrient Deficiency": "Though rich in minerals, clay soil may lack organic matter and drain poorly. This can cause root diseases and stunted growth in some plants.",
        "Recommended Fertilizers": "Apply compost and aged manure to improve structure and organic matter. Use gypsum to reduce compaction. Slow-release fertilizers prevent nutrient lockup.",
        "Tips to Improve Soil": "Work soil in dry conditions. Add organic matter consistently. Double digging and raised beds help aeration. Avoid walking on wet clay to prevent compaction."
    },
    "Loam": {
        "Crops Cultivatable": "Loamy soil is considered the best agricultural soil due to its balanced sand, silt, and clay composition. It supports nearly all crops such as maize, cotton, sugarcane, pulses, and vegetables.",
        "Nutrient Deficiency": "Rarely deficient, but overuse can deplete nitrogen and phosphorus. Careful rotation prevents micronutrient loss."
        ,"Recommended Fertilizers": "Compost and balanced NPK fertilizers maintain long-term fertility. Mulching prevents leaching.",
        "Tips to Improve Soil": "Avoid over-tilling to maintain structure. Rotate crops and include legumes. Maintain pH with lime or sulfur if necessary."
    },
    "Peat": {
        "Crops Cultivatable": "Peat soil, being high in organic matter and moisture, is excellent for root vegetables like carrots and turnips. It also supports legumes and brassicas when properly drained.",
        "Nutrient Deficiency": "Often acidic and low in minerals like iron, manganese, and molybdenum. Requires liming and trace element supplementation.",
        "Recommended Fertilizers": "Use lime to reduce acidity, and apply micronutrient-rich fertilizers. Add sand to improve drainage.",
        "Tips to Improve Soil": "Drain excess water through channels. Mix in sand or loam for structure. Regularly test pH and adjust."
    },
    "Chalk": {
        "Crops Cultivatable": "Chalky soils suit crops like barley, beans, spinach, and cabbage. They thrive in alkaline conditions and require adequate watering.",
        "Nutrient Deficiency": "Commonly lacks iron, manganese, and potassium. High pH can lead to nutrient lockout.",
        "Recommended Fertilizers": "Use acidifying fertilizers (ammonium sulfate), seaweed, and chelated iron sprays.",
        "Tips to Improve Soil": "Add compost and acidic organic matter. Avoid over-liming. Grow green manures to retain moisture."
    },
    "Alluvial Soil": {
        "Crops Cultivatable": "Highly fertile and found in river plains, supports paddy, wheat, sugarcane, maize, and pulses. Ideal for intensive cropping.",
        "Nutrient Deficiency": "May be deficient in nitrogen and phosphorus due to leaching in flood-prone zones.",
        "Recommended Fertilizers": "Apply nitrogen and phosphorus-based fertilizers like urea and DAP. Use vermicompost for sustainability.",
        "Tips to Improve Soil": "Use green manures post-harvest. Employ bunding and contour farming to reduce erosion."
    },
    "Black Cotton Soil (Regur)": {
        "Crops Cultivatable": "Excellent for cotton, soybean, sorghum, and sunflower. High moisture retention aids growth in dry regions.",
        "Nutrient Deficiency": "Deficient in nitrogen, phosphorus, and organic carbon. High in calcium and magnesium.",
        "Recommended Fertilizers": "Add nitrogenous and phosphatic fertilizers. Apply FYM and compost.",
        "Tips to Improve Soil": "Deep ploughing post-monsoon helps cracking soil structure. Use contour bunding."
    },
    "Red and Yellow Soil": {
        "Crops Cultivatable": "Suitable for millets, pulses, groundnut, and oilseeds. Common in eastern and central India.",
        "Nutrient Deficiency": "Low in nitrogen, phosphorus, and humus. Prone to erosion.",
        "Recommended Fertilizers": "Incorporate farmyard manure and green manure. Apply balanced NPK blends.",
        "Tips to Improve Soil": "Terracing and afforestation help prevent erosion. Use mulching for moisture."
    },
    "Laterite Soil": {
        "Crops Cultivatable": "Grows tea, coffee, cashew, rubber, and coconut in high rainfall areas.",
        "Nutrient Deficiency": "Low fertility due to leaching. Deficient in lime, potash, and phosphoric acid.",
        "Recommended Fertilizers": "Apply lime and organic compost. Supplement with potassium-rich fertilizers.",
        "Tips to Improve Soil": "Use raised beds to counter waterlogging. Replenish nutrients annually."
    }
}

# Function to predict soil type
def predict_soil_type(img_array):
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    soil_type = soil_types[predicted_index]
    confidence = prediction[predicted_index]
    return soil_type, confidence, prediction

# Generate a chart comparing soil nutrient probability and highlight predicted soil
def generate_comparison_chart(prediction_probs, detected_soil):
    fig, ax = plt.subplots()
    bars = ax.bar(soil_types, prediction_probs, color='gray')
    for i, soil in enumerate(soil_types):
        if soil == detected_soil:
            bars[i].set_color('green')
    ax.set_ylabel('Probability')
    ax.set_title('Soil Type Prediction Probabilities')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Mapping from soil type to major regions in India
soil_distribution = {
    "Sand": [("Rajasthan", 27.0238, 74.2179), ("Punjab", 31.1471, 75.3412)],
    "Silt": [("Uttar Pradesh", 26.8467, 80.9462)],
    "Clay": [("Tamil Nadu", 11.1271, 78.6569)],
    "Loam": [("Haryana", 29.0588, 76.0856)],
    "Peat": [("Kerala", 10.8505, 76.2711)],
    "Chalk": [("Himachal Pradesh", 31.1048, 77.1734)],
    "Alluvial Soil": [("Bihar", 25.0961, 85.3131), ("West Bengal", 22.9868, 87.8550)],
    "Black Cotton Soil (Regur)": [("Maharashtra", 19.7515, 75.7139), ("Madhya Pradesh", 22.9734, 78.6569)],
    "Red and Yellow Soil": [("Odisha", 20.9517, 85.0985), ("Chhattisgarh", 21.2787, 81.8661)],
    "Laterite Soil": [("Goa", 15.2993, 74.1240), ("Karnataka", 15.3173, 75.7139), ("Kerala", 10.8505, 76.2711)]
}

# Function to show map with soil distribution
def show_soil_map(soil_type):
    if soil_type not in soil_distribution:
        return
    st.markdown("### \U0001F4CD Major Regions with this Soil Type in India")
    india_map = folium.Map(location=[22.9734, 78.6569], zoom_start=5)
    for region, lat, lon in soil_distribution[soil_type]:
        folium.Marker([lat, lon], tooltip=f"{region} ({soil_type})", icon=folium.Icon(color='green')).add_to(india_map)
    st_folium(india_map, width=700)

# Sidebar Navigation
pages = ["Upload & Predict", "Crops Cultivatable", "Nutrient Deficiency", "Recommended Fertilizers", "Tips to Improve Soil"]
page = st.sidebar.radio("Navigation", pages)

# Session state initialization
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

# Upload & Predict Page
if page == "Upload & Predict":
    st.title("\U0001F33E Soil Health Monitoring System")
    img_source = st.radio("Select image input method:", ["Upload Image", "Take Photo"])

    if img_source == "Upload Image":
        image = st.file_uploader("Upload Soil Image", type=["jpg", "jpeg", "png"])
    else:
        image = st.camera_input("Take a photo")

    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.session_state.uploaded_image = image

        img = Image.open(image).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        soil_type, confidence, probabilities = predict_soil_type(img_array)
        st.session_state.prediction = (soil_type, confidence)
        st.session_state.probabilities = probabilities

        st.success(f"\U0001F9EA Detected Soil Type: {soil_type} ({confidence*100:.2f}% confidence)")

        show_soil_map(soil_type)
        generate_comparison_chart(probabilities, soil_type)

        if st.button("Download Report as PDF"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{soil_type.replace(' ', '_')}_{timestamp}.pdf"
            pdf_content = f"""
                <html>
                <head><meta charset='UTF-8'><style>h1{{text-align:center;}} h2{{color:#2E86C1}} p{{font-size:18px;}}</style></head>
                <body>
                <h1>{soil_type} Soil Report</h1>
                <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
            """
            for key, val in soil_info[soil_type].items():
                pdf_content += f"<h2>{key}</h2><p>{val}</p>"
            pdf_content += "</body></html>"

            tmp_path = os.path.join("./", filename)
            pdfkit.from_string(pdf_content, tmp_path, configuration=pdfkit_config)

            with open(tmp_path, "rb") as f:
                st.download_button("\U0001F4C4 Download PDF", f, file_name=filename)

# Subtopic Pages
else:
    st.title(f"\U0001F4DA {page}")
    if st.session_state.prediction:
        soil_type = st.session_state.prediction[0]
        st.markdown(f"### For <u>{soil_type}</u> soil:", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px;'>{soil_info.get(soil_type, {}).get(page, 'No information available.')}</p>", unsafe_allow_html=True)
    else:
        st.warning("Please upload and predict a soil image first from the Upload & Predict tab.")
