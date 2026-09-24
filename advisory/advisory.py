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

def format_advisory_text(advisory: dict, severity: str = None) -> str:
    """Formats the advisory dict into a single readable paragraph for translation/TTS."""
    treatment = advisory.get('treatment', 'Unknown')
    
    # Override treatment if severity-specific text exists
    if severity and 'treatment_severity' in advisory:
        if severity in advisory['treatment_severity']:
            treatment = advisory['treatment_severity'][severity]
            
    return f"Cause: {advisory.get('cause', 'Unknown')}\n\nTreatment: {treatment}\n\nPrevention: {advisory.get('prevention', 'Unknown')}"
