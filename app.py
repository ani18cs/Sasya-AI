import streamlit as st
from PIL import Image
from language.sarvam_client import translate_to_kannada, kannada_text_to_speech

st.set_page_config(page_title="Sasya AI", page_icon="🌿", layout="centered")

st.title("Sasya AI — ಸಸ್ಯ AI 🌿")
st.write("Upload a leaf photo to get an instant diagnosis in Kannada.")

uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded leaf", use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        # TODO: wire up actual model
        dummy_prediction = "Tomato___Early_blight"
        dummy_confidence = 0.92
        
        # TODO: wire up actual grad-cam
        with col2:
            st.image(image, caption="Grad-CAM Explainability Heatmap (Mock)", use_container_width=True)
            
    st.success(f"**Diagnosis:** {dummy_prediction} ({dummy_confidence:.1%} confidence)")
    
    st.subheader("Treatment Advisory")
    # TODO: wire up actual advisory JSON
    dummy_advisory_en = "Remove and destroy infected leaves. Apply a copper-based fungicide. Avoid overhead watering."
    
    # Translate and TTS
    with st.spinner("Translating to Kannada..."):
        kannada_text = translate_to_kannada(dummy_advisory_en)
        audio_path = kannada_text_to_speech(kannada_text)
        
    st.write(f"**ಕನ್ನಡ (Kannada):** {kannada_text}")
    st.audio(audio_path, format="audio/wav")
