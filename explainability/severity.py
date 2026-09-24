def compute_severity(grayscale_cam, threshold=0.5):
    """
    grayscale_cam: the 2D numpy array already produced by our existing 
    Grad-CAM call in explainability/gradcam.py (values 0-1).
    Returns: (affected_fraction: float, label: str)
    """
    affected_fraction = (grayscale_cam > threshold).mean()
    if affected_fraction < 0.15:
        label = "Mild"
    elif affected_fraction < 0.40:
        label = "Moderate"
    else:
        label = "Severe"
    return affected_fraction, label
