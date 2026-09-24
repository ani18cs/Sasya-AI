import json
import os
from language.sarvam_client import get_client, kannada_text_to_speech, translate_to_kannada

SYSTEM_PROMPT = """You are answering a farmer's follow-up question about a 
specific crop disease diagnosis. You may ONLY use the diagnosis and advisory 
context provided below — do not add any treatment, dosage, or chemical 
recommendation not already present in this context. If the farmer's question 
cannot be answered from this context, say so clearly and recommend they 
consult a local agricultural officer. Do not guess or fabricate agricultural 
advice under any circumstances."""

def transcribe_kannada_question(audio_input) -> str:
    """
    Transcribes Kannada audio into text using Sarvam's Saaras STT API.
    audio_input can be a file path or file-like object.
    """
    client = get_client()
    if not client:
        raise Exception("Sarvam client not initialized.")
        
    try:
        # If audio_input is a path, open it. Otherwise assume it's file-like (from st.audio_input)
        if isinstance(audio_input, str):
            f = open(audio_input, "rb")
            close_f = True
        else:
            f = audio_input
            f.seek(0)
            close_f = False
            
        # Call the transcription API
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            mode="transcribe"
        )
        
        if close_f:
            f.close()
            
        # The response structure typically has a 'transcript' or similar field. 
        # For Sarvam, it's response.transcript
        return response.transcript
    except Exception as e:
        raise Exception(f"STT failed: {str(e)}")

def translate_to_english(kannada_text: str) -> str:
    """Translates Kannada text to English."""
    client = get_client()
    if not client:
        return kannada_text # Dummy fallback
        
    response = client.text.translate(
        input=kannada_text,
        source_language_code="kn-IN",
        target_language_code="en-IN"
    )
    return response.translated_text

def answer_followup(question_kn: str, advisory_context: dict) -> str:
    """
    Translates the question, asks the LLM to answer using ONLY the context, 
    and returns the English answer.
    """
    client = get_client()
    if not client:
        raise Exception("Sarvam client not initialized.")
        
    try:
        # Translate to English
        question_en = translate_to_english(question_kn)
        
        # Format the context strictly
        context_str = json.dumps(advisory_context, indent=2)
        
        # Call the chat model
        response = client.chat.completions(
            model="sarvam-105b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n\nContext:\n" + context_str},
                {"role": "user", "content": question_en}
            ]
        )
        
        answer_text_en = response.choices[0].message.content
        return answer_text_en
    except Exception as e:
        raise Exception(f"Chat completion failed: {str(e)}")

def speak_answer(answer_text_en: str, output_path: str = "qa_audio.wav") -> str:
    """
    Translates English answer to Kannada and synthesizes speech.
    Returns the path to the generated audio file.
    """
    try:
        answer_kn = translate_to_kannada(answer_text_en)
        # Call existing TTS logic
        return kannada_text_to_speech(answer_kn, output_path=output_path)
    except Exception as e:
        raise Exception(f"TTS failed: {str(e)}")
