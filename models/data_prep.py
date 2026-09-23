"""
Script to download, clean, and consolidate datasets from Kaggle, HuggingFace, etc.
"""
import os
import subprocess
import shutil

def download_kaggle_dataset(dataset_name, download_path):
    print(f"Downloading {dataset_name} from Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", download_path, "--unzip"], check=True)
        print("Download and unzip complete.")
    except Exception as e:
        print(f"Error downloading {dataset_name}. Make sure Kaggle API is configured correctly. {e}")

def prepare_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("Step 1: Downloading PlantVillage Dataset...")
    download_kaggle_dataset("vipoooool/new-plant-diseases-dataset", data_dir)
    
    print("Step 2: Note - Placeholders for additional datasets (PlantDoc, FieldPlant, Mendeley)")
    print("These will require custom auth (HuggingFace token, Roboflow API key) when running locally.")
    
    # TODO: Implement consolidation logic to merge folders into a single train/val split structure
    # taking into account different image resolutions and formats.

if __name__ == "__main__":
    prepare_data()
