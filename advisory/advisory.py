import json
import os

def load_advisory_db():
    db_path = os.path.join(os.path.dirname(__file__), "advisory_db.json")
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_advisory(disease_class: str) -> dict:
    """
    Returns the advisory information for a given disease class.
    Falls back to a default message if the class is not found.
    """
    db = load_advisory_db()
    
    # Check exact match
    if disease_class in db:
        return db[disease_class]
        
    # Fallback
    return {
        "cause": "Unknown",
        "treatment": "Please consult a local agricultural officer.",
        "prevention": "Maintain standard crop hygiene and monitor regularly."
    }

def format_advisory_text(advisory: dict) -> str:
    """Formats the advisory dict into a single readable paragraph for translation/TTS."""
    return f"Cause: {advisory.get('cause', 'Unknown')}. Treatment: {advisory.get('treatment', 'Unknown')}. Prevention: {advisory.get('prevention', 'Unknown')}."
