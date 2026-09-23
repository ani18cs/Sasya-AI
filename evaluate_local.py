import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

def build_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data", "filtered")
    val_dir = os.path.join(data_dir, "valid")
    
    if not os.path.exists(val_dir):
        # Fallback to the original dataset layout if filtered doesn't exist
        val_dir = os.path.join(os.path.dirname(__file__), "data", "valid")
        
    print(f"Validation directory: {val_dir}")
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(val_dir, val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    num_classes = len(val_dataset.classes)
    print(f"Found {num_classes} classes: {val_dataset.classes}")
    
    model = build_model(num_classes).to(device)
    weights_path = os.path.join(os.path.dirname(__file__), "models", "resnet50_sasya.pth")
    print(f"Loading weights from: {weights_path}")
    
    # Try loading the weights
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Failed to load standard state_dict, trying without weights_only: {e}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes, digits=4))

if __name__ == "__main__":
    evaluate()
