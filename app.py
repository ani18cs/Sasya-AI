import streamlit as st
from PIL import Image

st.title("Sasya AI — ಸಸ್ಯ AI")
st.write("Upload a leaf photo to get an instant diagnosis in Kannada.")

uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded leaf")
    
    st.info("Diagnosis model, Explainability heatmap, and Kannada advisory will appear here.")
    # TODO: run through model → gradcam → advisory → sarvam translate → sarvam TTS
