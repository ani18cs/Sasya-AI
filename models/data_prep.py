"""
Script to download, clean, and consolidate datasets from Kaggle, HuggingFace, etc.
"""
import os
import subprocess
import shutil

def download_kaggle_dataset(dataset_name, download_path):
    print(f"Downloading {dataset_name} from Kaggle...")
    try:
        subprocess.run(["python", "-m", "kaggle", "datasets", "download", "-d", dataset_name, "-p", download_path, "--unzip"], check=True)
        print("Download and unzip complete.")
    except Exception as e:
        print(f"Error downloading {dataset_name}. Make sure Kaggle API is configured correctly. {e}")

def prepare_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("Step 1: Downloading PlantVillage Dataset...")
    download_kaggle_dataset("vipoooool/new-plant-diseases-dataset", data_dir)
    
    print("Step 2: Filtering dataset for Maize, Tomato, Grape, and Potato...")
    # The Kaggle unzip usually creates a nested folder structure
    # Expected: data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/...
    base_extract = os.path.join(data_dir, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)")
    
    target_crops = ["Corn", "Tomato", "Grape", "Potato"]
    
    for split in ["train", "valid"]:
        src_split = os.path.join(base_extract, split)
        dst_split = os.path.join(data_dir, split)
        os.makedirs(dst_split, exist_ok=True)
        
        if os.path.exists(src_split):
            for class_name in os.listdir(src_split):
                if any(crop in class_name for crop in target_crops):
                    src_class = os.path.join(src_split, class_name)
                    dst_class = os.path.join(dst_split, class_name)
                    if not os.path.exists(dst_class):
                        shutil.move(src_class, dst_class)
                        
    print("Step 3: Verifying final class list and image counts...")
    for split in ["train", "valid"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        print(f"\n--- {split.upper()} SET ---")
        classes = sorted(os.listdir(split_dir))
        print(f"Total classes: {len(classes)}")
        for cls in classes:
            count = len(os.listdir(os.path.join(split_dir, cls)))
            warning = " (⚠️ LOW COUNT)" if count < 200 else ""
            print(f"  {cls}: {count} images{warning}")
                        
    print("\nData prep complete. Filtered classes are in data/train and data/valid.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    prepare_data()
