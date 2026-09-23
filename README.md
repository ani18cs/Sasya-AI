# Sasya AI

Sasya AI is an explainable, Kannada-speaking crop disease diagnosis app for Karnataka farmers.

## Architecture
1. **Classifier:** ResNet-50 CNN for disease diagnosis.
2. **Explainability Layer:** Grad-CAM for visual trust.
3. **Advisory Engine:** Deterministic rule-based lookup for verified treatment advice (zero hallucination).
4. **Language Layer:** Sarvam AI for English to Kannada translation and TTS.

## Setup
1. `pip install -r requirements.txt`
2. Set up `.env` with `SARVAM_API_KEY`.
3. Run `streamlit run app.py`
