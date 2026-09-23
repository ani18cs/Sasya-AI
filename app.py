import streamlit as st
from PIL import Image
import torch
import numpy as np
import os
from torchvision import transforms

from language.sarvam_client import translate_to_kannada, kannada_text_to_speech
from advisory.advisory import get_advisory, format_advisory_text
from models.train import build_model
from explainability.gradcam import get_gradcam_overlay

st.set_page_config(page_title="Sasya AI", page_icon="🌿", layout="centered")

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
st.write("Upload a leaf photo to get an instant diagnosis in Kannada.")

uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded leaf", use_container_width=True)
    
    with st.spinner("Analyzing image..."):
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
            
            # Grad-CAM
            target_layer = model.layer4[-1]
            rgb_img = np.array(image.resize((224, 224)))
            if rgb_img.shape[-1] == 4: # Handle RGBA
                rgb_img = rgb_img[..., :3]
            
            cam_image = get_gradcam_overlay(model, input_tensor, rgb_img, target_layer)
            
            with col2:
                st.image(cam_image, caption="Grad-CAM Explainability Heatmap", use_container_width=True)
                
        else:
            # Fallback to dummy data if model isn't trained yet
            prediction = "Tomato___Early_blight"
            confidence = 0.92
            with col2:
                st.image(image, caption="Grad-CAM Explainability Heatmap (Mock - Model not found)", use_container_width=True)
            
    st.success(f"**Diagnosis:** {prediction} ({confidence:.1%} confidence)")
    
    st.subheader("Treatment Advisory")
    # Fetch advisory from JSON DB
    advisory_data = get_advisory(prediction)
    dummy_advisory_en = format_advisory_text(advisory_data)
    
    st.write(f"**Cause:** {advisory_data.get('cause', 'Unknown')}")
    st.write(f"**Treatment:** {advisory_data.get('treatment', 'Unknown')}")
    st.write(f"**Prevention:** {advisory_data.get('prevention', 'Unknown')}")
    
    # Translate and TTS
    with st.spinner("Translating to Kannada..."):
        kannada_text = translate_to_kannada(dummy_advisory_en)
        audio_path = kannada_text_to_speech(kannada_text)
        
    st.write(f"**ಕನ್ನಡ (Kannada):** {kannada_text}")
    st.audio(audio_path, format="audio/wav")
