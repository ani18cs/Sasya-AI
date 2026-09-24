# Sasya AI: End-to-End Walkthrough

## 1. Background & Scope
This session focused on **baseline model training and end-to-end integration**. Specifically, we:
- Fixed bugs with the Sarvam TTS API integration (base64 audio decoding).
- Evaluated and processed the Kaggle "New Plant Diseases Dataset" to create a balanced subset of 21 core classes.
- Created and executed a Kaggle Notebook to train a PyTorch ResNet-50 transfer-learning baseline.
- Integrated the trained weights back into the local `app.py` for headless and UI inference.

## 2. Changes Made
- **Kaggle Kernel**: Authored and pushed `baseline_training.ipynb` which uses the Kaggle API to train ResNet-50.
- **Model Integration**: The model weights (`resnet50_sasya.pth`) were placed inside the `models/` directory after manual download.
- **`app.py`**: Updated the UI logic to seamlessly link PyTorch ResNet-50 inference -> Grad-CAM explainability -> Advisory JSON matching -> Sarvam Kannada TTS.

## 3. Evaluation & Validation Results

> [!WARNING]
> **NETWORK FAILURE**: The automated local evaluation failed because the agent's environment was unable to download the `torch` PyPI package due to network connection aborts (`[WinError 10053]`), the exact same issue that blocked the automated Kaggle weights download.

Since you have the weights locally, you can **run the evaluation scripts yourself** to verify the end-to-end integration:

1. **Verify Metrics** (Accuracy, Precision, Recall, F1):
   Run the evaluation script to calculate metrics on the validation dataset:
   ```bash
   python evaluate_local.py
   ```

2. **Verify End-to-End Pipeline**:
   Run the integration script which picks a validation image, runs ResNet-50, generates the Grad-CAM heatmap, fetches the advisory, translates it, and calls Sarvam TTS.
   ```bash
   python integration_test.py
   ```

## 4. Next Steps
Once you verify the outputs of the evaluation scripts, we are completely finished with Part 1. We can then move on to **Part 2** (defining the parallel workstreams for the other team members).
