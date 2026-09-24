import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import base64

from language.sarvam_client import translate_to_kannada, kannada_text_to_speech
from advisory.advisory import get_advisory, format_advisory_text
from models.train import build_model
from explainability.gradcam import get_gradcam_overlay

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def main():
    print("Loading model...")
    weights_path = os.path.join(os.path.dirname(__file__), "models", "resnet50_sasya.pth")
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
    
    model = build_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    # Pick a random validation image
    val_dir = os.path.join(os.path.dirname(__file__), "data", "filtered", "valid")
    # if it doesn't exist, fallback to original
    if not os.path.exists(val_dir):
        val_dir = os.path.join(os.path.dirname(__file__), "data", "valid")
        
    sample_class = "Tomato___Early_blight"
    sample_dir = os.path.join(val_dir, sample_class)
    sample_image_name = os.listdir(sample_dir)[0]
    sample_image_path = os.path.join(sample_dir, sample_image_name)
    
    print(f"Testing with image: {sample_image_path}")
    image = Image.open(sample_image_path)
    
    print("Running inference...")
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    prediction = class_names[predicted_idx.item()]
    confidence = confidence.item()
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")
    print(f"Expected: {sample_class}")
    
    print("Generating Grad-CAM...")
    target_layer = model.layer4[-1]
    rgb_img = np.array(image.resize((224, 224)))
    if rgb_img.shape[-1] == 4:
        rgb_img = rgb_img[..., :3]
    
    cam_image = get_gradcam_overlay(model, input_tensor, rgb_img, target_layer)
    cam_path = os.path.join(os.path.dirname(__file__), "test_cam_output.png")
    Image.fromarray(cam_image).save(cam_path)
    print(f"Grad-CAM saved to: {cam_path}")
    
    print("Fetching advisory...")
    advisory_data = get_advisory(prediction)
    dummy_advisory_en = format_advisory_text(advisory_data)
    print(f"Advisory text: {dummy_advisory_en}")
    
    print("Testing translation and TTS...")
    from dotenv import load_dotenv
    load_dotenv()
    kannada_text = translate_to_kannada(dummy_advisory_en)
    print(f"Kannada text: {kannada_text}")
    
    audio_path = kannada_text_to_speech(kannada_text)
    print(f"Audio saved to: {audio_path}")
    
    print("Integration test passed end-to-end!")

if __name__ == "__main__":
    main()
