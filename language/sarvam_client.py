"""
Sarvam AI client for Kannada translation and text-to-speech.
"""
import os
from sarvamai import SarvamAI

def get_client():
    return SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))

def translate_to_kannada(english_text: str) -> str:
    client = get_client()
    # Check live docs for exact method signature
    response = client.text.translate(
        input=english_text,
        source_language_code="en-IN",
        target_language_code="kn-IN"
    )
    return response.translated_text

def kannada_text_to_speech(kannada_text: str, output_path="advisory_audio.wav"):
    client = get_client()
    # Check live docs for exact method signature
    response = client.text_to_speech.convert(
        text=kannada_text,
        target_language_code="kn-IN",
        model="bulbul:v3"
    )
    with open(output_path, "wb") as f:
        f.write(response.audio) 
    return output_path
