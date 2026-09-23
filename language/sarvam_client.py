"""
Sarvam AI client for Kannada translation and text-to-speech.
"""
import os

try:
    from sarvamai import SarvamAI
except ImportError:
    SarvamAI = None

def get_client():
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key or not SarvamAI:
        return None
    return SarvamAI(api_subscription_key=api_key)

def translate_to_kannada(english_text: str) -> str:
    client = get_client()
    if not client:
        return f"[DUMMY KANNADA TRANSLATION]: {english_text}"
        
    response = client.text.translate(
        input=english_text,
        source_language_code="en-IN",
        target_language_code="kn-IN"
    )
    return response.translated_text

def kannada_text_to_speech(kannada_text: str, output_path="advisory_audio.wav"):
    client = get_client()
    if not client:
        # Create a dummy empty wav file to unblock the UI
        with open(output_path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
        return output_path
        
    response = client.text_to_speech.convert(
        text=kannada_text,
        target_language_code="kn-IN",
        model="bulbul:v3"
    )
    with open(output_path, "wb") as f:
        f.write(response.audio) 
    return output_path
