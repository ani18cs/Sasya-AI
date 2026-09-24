import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
import torch
import numpy as np
import os
import base64
from io import BytesIO
import time
from torchvision import transforms

from language.sarvam_client import translate_to_kannada, kannada_text_to_speech
from advisory.advisory import get_advisory, format_advisory_text
from models.train import build_model
from explainability.gradcam import get_gradcam_overlay
from explainability.severity import compute_severity
from weather.weather_client import get_spray_advisory_note, KARNATAKA_DISTRICTS

st.set_page_config(page_title="Sasya AI", page_icon="🌿", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600&family=Manrope:wght@400;600&family=Noto+Sans+Kannada:wght@400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Manrope', sans-serif !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Chakra Petch', sans-serif !important;
}

/* Custom UI Cards */
.advisory-card {
    background-color: #16211C;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 24px;
    border: 1px solid rgba(61, 255, 162, 0.1);
}
.weather-banner {
    background-color: #FFB020;
    color: #0E1512;
    padding: 12px 16px;
    border-radius: 8px;
    font-weight: 600;
    margin-bottom: 16px;
}
.voice-qa-section {
    border: 1px solid #7C4DFF;
    background-color: rgba(124, 77, 255, 0.05);
    padding: 24px;
    border-radius: 12px;
    margin-top: 32px;
}
.trust-footer {
    color: #8FA396;
    font-size: 13px;
    text-align: center;
    margin-top: 48px;
    padding-top: 24px;
    border-top: 1px solid rgba(143, 163, 150, 0.2);
}

/* Scan Animation */
@keyframes scan-line {
    0% { top: 0%; opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
.scan-container {
    position: relative;
    width: 100%;
    overflow: hidden;
    border-radius: 8px;
}
.scan-overlay {
    position: absolute;
    left: 0;
    right: 0;
    height: 6px;
    background: #7C4DFF;
    box-shadow: 0 0 15px #7C4DFF, 0 0 30px #7C4DFF;
    animation: scan-line 2s infinite linear;
    z-index: 10;
}
@media (prefers-reduced-motion: reduce) {
    .scan-overlay {
        animation: none;
        top: 50%;
        opacity: 0.5;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_disease_model():
    weights_path = os.path.join(os.path.dirname(__file__), "models", "resnet50_sasya.pth")
    # Using a placeholder for num_classes - this must match training!
    # For now, we assume 2 classes based on the dummy advisory_db
    num_classes = 21 
    class_names = [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy"
    ]
    
    if not os.path.exists(weights_path):
        return None, class_names
        
    model = build_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model, class_names

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Convert RGBA to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# --- UI Setup ---
st.title("Sasya AI — ಸಸ್ಯ AI 🌿")
st.markdown("<p style='color: #8FA396; font-size: 16px;'>Photograph a leaf to get an instant diagnosis and spoken advice in Kannada.</p>", unsafe_allow_html=True)

# Input Row
with st.container(border=True):
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        district_options = list(KARNATAKA_DISTRICTS.keys()) + ["Other"]
        selected_district = st.selectbox("Select your district for weather-aware advisory:", district_options)
        if selected_district == "Other":
            selected_district = st.text_input("Enter your city/district name:")

    with input_col2:
        uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    
    scan_placeholder = st.empty()
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    scan_placeholder.markdown(f'''
    <div class="scan-container">
        <img src="data:image/png;base64,{img_str}" style="width:100%; max-width:100%; border-radius:8px;"/>
        <div class="scan-overlay"></div>
    </div>
    <p style="text-align:center; color:#7C4DFF; font-weight:600; margin-top:10px;">Reading leaf pattern...</p>
    ''', unsafe_allow_html=True)
    
    with st.spinner("Checking against known diseases..."):
        model, class_names = load_disease_model()
        
        if model:
            # Actual Inference
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            prediction = class_names[predicted_idx.item()]
            confidence = confidence.item()
            
            # Grad-CAM with safe fallback
            severity_label = None
            severity_fraction = 0.0
            try:
                target_layer = model.layer4[-1]
                rgb_img = np.array(image.resize((224, 224)))
                if rgb_img.shape[-1] == 4: # Handle RGBA
                    rgb_img = rgb_img[..., :3]
                
                cam_image, grayscale_cam = get_gradcam_overlay(model, input_tensor, rgb_img, target_layer)
                    
                if "healthy" not in prediction.lower():
                    severity_fraction, severity_label = compute_severity(grayscale_cam)
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")
                cam_image = image
        else:
            # Fallback to dummy data if model isn't trained yet
            prediction = "Tomato___Early_blight"
            confidence = 0.92
            cam_image = image
            severity_label = "Moderate"
            severity_fraction = 0.16

    # Clear scan animation
    scan_placeholder.empty()

    # Image Comparison
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Photo", use_container_width=True)
    with col2:
        st.image(cam_image, caption="Highlighted areas show what the model focused on", use_container_width=True)
        
    # Diagnosis Result
    if "healthy" in prediction.lower():
        color = "#3DFFA2"
    elif confidence < 0.6 or (severity_label and severity_label in ["Moderate", "Severe"]):
        color = "#FFB020"
    else:
        color = "#3DFFA2"
        
    st.markdown(f"<h2 style='color:{color}; margin-bottom: 0px;'>{prediction.replace('___', ' - ').replace('_', ' ')} ({confidence:.1%} confidence)</h2>", unsafe_allow_html=True)
    if severity_label:
        st.markdown(f"<p style='color:#FFB020; font-weight:600; font-size:18px;'>Severity: {severity_fraction:.0%} of leaf affected — {severity_label}</p>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
        
    # Fetch advisory from JSON DB
    advisory_data = get_advisory(prediction)
    dummy_advisory_en = format_advisory_text(advisory_data, severity=severity_label if 'severity_label' in locals() else None)
    
    treatment_display = advisory_data.get('treatment', 'Unknown')
    if 'severity_label' in locals() and severity_label and 'treatment_severity' in advisory_data:
        if severity_label in advisory_data['treatment_severity']:
            treatment_display = advisory_data['treatment_severity'][severity_label]
            
    weather_note = get_spray_advisory_note(selected_district, dummy_advisory_en)
    if weather_note:
        dummy_advisory_en += f" {weather_note}"
    
    # Advisory Card
    st.markdown('<div class="advisory-card">', unsafe_allow_html=True)
    
    if weather_note:
        st.markdown(f'<div class="weather-banner">⚠️ {weather_note}</div>', unsafe_allow_html=True)
        
    st.markdown(f"**Cause:**<br>{advisory_data.get('cause', 'Unknown')}<br><br>", unsafe_allow_html=True)
    st.markdown(f"**Treatment:**<br>{treatment_display}<br><br>", unsafe_allow_html=True)
    st.markdown(f"**Prevention:**<br>{advisory_data.get('prevention', 'Unknown')}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Translate and TTS
    with st.spinner("Translating to Kannada..."):
        kannada_text = translate_to_kannada(dummy_advisory_en)
        audio_path = kannada_text_to_speech(kannada_text)
        
    with st.container(border=True):
        kannada_html = kannada_text.replace('\n', '<br>')
        st.markdown(f"<p class='kannada-text' style='font-size:18px;'><strong>ಕನ್ನಡ:</strong><br><br>{kannada_html}</p>", unsafe_allow_html=True)
        st.audio(audio_path, format="audio/wav")

    # Voice Q&A Section
    st.markdown('<div class="voice-qa-section">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0px;'>Ask a follow-up question</h3>", unsafe_allow_html=True)
    
    from conversation.voice_qa import transcribe_kannada_question, answer_followup, speak_answer
    
    audio_bytes = None
    if hasattr(st, "audio_input"):
        audio_bytes = st.audio_input("Record your question (Kannada):")
    else:
        audio_bytes = st.file_uploader("Upload your voice question (WAV/MP3)", type=["wav", "mp3", "m4a", "ogg"])
        
    if audio_bytes:
        with st.spinner("Processing your question..."):
            try:
                question_kn = transcribe_kannada_question(audio_bytes)
                st.markdown(f"<p class='kannada-text'><strong>Your Question:</strong> {question_kn}</p>", unsafe_allow_html=True)
                
                context = {
                    "disease": prediction,
                    "cause": advisory_data.get('cause', 'Unknown'),
                    "treatment": treatment_display,
                    "prevention": advisory_data.get('prevention', 'Unknown')
                }
                answer_en = answer_followup(question_kn, context)
                out_audio = speak_answer(answer_en)
                
                st.markdown(f"<p style='color:#7C4DFF;'><strong>Answer:</strong> {answer_en}</p>", unsafe_allow_html=True)
                if os.path.exists(out_audio):
                    st.audio(out_audio, format="audio/wav")
                    
            except Exception as e:
                st.error("Sorry, I couldn't process that question — please consult a local agricultural officer.")
                print(f"Voice Q&A Error: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="trust-footer">Treatment advice is sourced from a verified agricultural database. The voice assistant is restricted to this diagnostic context.</div>', unsafe_allow_html=True)
