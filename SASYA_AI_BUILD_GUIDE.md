# Sasya AI — Complete Build Guide & Knowledge Base

**Project**: Sasya AI — Explainable, Kannada-speaking crop disease diagnosis for Karnataka farmers
**Team size**: 3
**Deadline**: MVP presentable by Sep 25, 2026 (Bengaluru Tech Summit 2026 college selection round, theme: "AI & Beyond")
**Build environment**: Google Antigravity (agent-first IDE, Gemini 3-based, supports Claude Sonnet)

This document is the single source of truth for the project. It is written so that:
1. All three team members can work in parallel without blocking each other.
2. It can be pasted into Antigravity (or any AI coding agent) as a spec to generate/verify code against.
3. Every design decision — especially around hallucination and trust — is explicit, not assumed.

---

## 1. Project Overview

Sasya AI lets a farmer photograph a diseased leaf and receive, within seconds:
1. A **disease diagnosis** with a confidence score (computer vision model).
2. A **visual explanation** of *why* the model reached that diagnosis (Grad-CAM heatmap).
3. A **treatment recommendation** (deterministic advisory lookup — not free-generated text).
4. The recommendation **spoken aloud in Kannada** (Sarvam AI translate + TTS).

Scope for the summit MVP: **static image upload only**. Live camera feed is explicitly future scope — do not attempt it before the core pipeline is stable.

### Why this design order matters
Each layer depends on the one before it. Build and validate top-to-bottom, not all four in parallel from day one — see Section 10 for how to parallelize safely.

---

## 2. High-Level Architecture

```
[Leaf photo] 
      │
      ▼
[Preprocessing: resize, normalize]
      │
      ▼
[CNN Classifier — ResNet-50, transfer learning]
      │
      ├──► [Prediction + confidence score]
      │
      ▼
[Grad-CAM explainability layer]
      │  (highlights pixels that drove the prediction)
      ▼
[Advisory Engine — deterministic lookup table]
      │  disease_class → { cause, treatment, prevention } in English
      ▼
[Sarvam Translate API — English → Kannada text]
      │
      ▼
[Sarvam Bulbul TTS — Kannada text → Kannada audio]
      │
      ▼
[UI: shows image + heatmap + confidence + Kannada text + Kannada audio playback]
```

**Critical design decision**: the Advisory Engine is a **rule-based lookup table, not an LLM call**. See Section 8 for why — this is the core anti-hallucination decision for the project and you should be ready to explain it to judges as a deliberate trust/safety choice, not a shortcut.

---

## 3. Reference Repositories

Two public repos are your architectural reference. Clone both, but build your working pipeline from Repo A. Do not literally merge the two codebases — pull ideas and dataset scope from Repo B into Repo A's cleaner structure.

### Repo A (primary base): `bPavan16/Plant-Disease-classification-using-CNN`
- URL: https://github.com/bPavan16/Plant-Disease-classification-using-CNN
- Architecture: ResNet-50 (PyTorch), transfer learning, ~82% validation accuracy reported.
- Why use this as the base: small, clean, easy for all three of you to actually understand and modify within a day. This is what you'll retrain, extend, and wrap in a UI.
- What to take from it: the training script structure, the ResNet-50 fine-tuning approach, the data loading pipeline.

**Clone command:**
```bash
git clone https://github.com/bPavan16/Plant-Disease-classification-using-CNN.git sasya-ai-base
cd sasya-ai-base
```

### Repo B (reference only): `Pallav7533/Dr.Plant`
- URL: https://github.com/Pallav7533/Dr.Plant
- Architecture: compares custom CNN, VGG16, and ResNet34, trained on 38 disease classes across 14 plants (PlantVillage dataset), reports 98.42% test accuracy.
- Why reference it: it shows you the full 38-class scope and the accuracy ceiling achievable on PlantVillage, and its README documents the dataset structure clearly.
- What to take from it: **dataset scope** (which PlantVillage classes exist), and optionally compare its accuracy numbers against your own retrained model as a sanity check.
- Do **not** copy its architecture code wholesale — you're building on Repo A's simpler pipeline.

**Clone command (reference only, don't build directly on this):**
```bash
git clone https://github.com/Pallav7533/Dr.Plant.git reference-drplant
```

### What "making it your own" means here (say this explicitly to judges)
- You are retraining on your **own selected subset of classes** relevant to Karnataka (see Section 4).
- You are adding **three layers neither repo has**: Grad-CAM explainability, the advisory engine, and the Kannada voice layer via Sarvam.
- You will report your **own accuracy/confusion-matrix numbers** from your own training run, not the original repos' numbers.
- Attribution: credit both repos by name/link in your README and final slide. This is normal practice and reads as more credible than hiding it.

---

## 4. Dataset

### Primary dataset: PlantVillage
- Source: https://www.kaggle.com/vipoooool/new-plant-diseases-dataset (Kaggle, requires free Kaggle account)
- ~87,000 images, 38 classes across 14 crop species, healthy + diseased leaf images.
- Download via Kaggle API (faster than browser download):
```bash
pip install kaggle --break-system-packages
# place kaggle.json (from kaggle.com/settings) in ~/.kaggle/
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d data/
```

### Karnataka-relevant crop subset — use these classes only
Do not train on all 38 classes. Filter down to crops that are both genuinely significant to Karnataka agriculture and well-represented in PlantVillage:

| Crop | Why relevant to Karnataka | Classes available in PlantVillage |
|---|---|---|
| Maize (corn) | Karnataka is a top Indian producer | Healthy, Common Rust, Gray Leaf Spot, Northern Leaf Blight |
| Tomato | Major horticultural crop statewide | Healthy + 9 disease classes (largest class group in dataset) |
| Grape | Major crop in Vijayapura/Bagalkot belt | Healthy, Black Rot, Esca, Leaf Blight |
| Potato | Widely grown, high disease sensitivity | Healthy, Early Blight, Late Blight |

This gives you **~15–18 classes**, a manageable and defensible scope you can explain fully in Q&A.

### Optional stretch dataset: Coffee leaf disease
- Karnataka grows ~70% of India's coffee (Kodagu, Chikmagalur) — strong regional narrative if time allows.
- Search Kaggle for "coffee leaf disease dataset" — several open options exist (e.g. datasets covering rust, miner, and healthy coffee leaves).
- Only add this if the core 4-crop pipeline is fully working first. Treat as a bonus class, not a blocker.

### Data split
- 80% train / 10% validation / 10% test, stratified by class.
- Use `torchvision.datasets.ImageFolder` + `random_split`, or `sklearn.model_selection.train_test_split` on file paths if you want stratification control.

---

## 5. Model Architecture & Training

### Base architecture
ResNet-50, pretrained on ImageNet, fine-tuned on your Karnataka crop subset (transfer learning — do not train from scratch, you don't have time or data volume for that).

```python
import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V2")
    # Freeze early layers, fine-tune later layers + new classifier head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

### Training config (starting point — tune if time allows)
- Optimizer: Adam, lr=1e-4 for the new head, 1e-5 for unfrozen layer4
- Loss: CrossEntropyLoss
- Batch size: 32 (reduce to 16 if training on CPU/limited Colab GPU)
- Epochs: 10–15 is enough for transfer learning to converge on this dataset size
- Data augmentation: random horizontal flip, rotation (±15°), color jitter — leaves photographed by farmers will have inconsistent lighting/angle, so augment for that

### Where to train
- **Google Colab (free tier, T4 GPU)** — fastest path given your time constraint. Upload the filtered dataset subset (not the full 87k images) to keep upload/training time manageable.
- If using Antigravity locally and it doesn't have GPU access, train in Colab and pull the trained `.pth` weights file back into your Antigravity project.

### Evaluation — report these, not just accuracy
- Overall accuracy
- Per-class precision/recall/F1 (some diseases will have fewer images — know which ones)
- Confusion matrix (screenshot this for your slides — it's a strong "we did real ML" artifact)
- A held-out test on 5–10 images you personally photograph or source separately from PlantVillage, to sanity-check real-world generalization beyond the clean dataset

```python
from sklearn.metrics import classification_report, confusion_matrix
# after getting y_true, y_pred on test set:
print(classification_report(y_true, y_pred, target_names=class_names))
```

---

## 6. Explainability Layer — Grad-CAM

Grad-CAM highlights which pixels of the input image most influenced the model's prediction, overlaid as a heatmap. This is your "trustworthy AI" layer — genuinely valuable for the pitch, not decoration.

Use `pytorch-grad-cam` (well-maintained library, saves you from implementing Grad-CAM from scratch):
```bash
pip install grad-cam --break-system-packages
```

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

def get_gradcam_overlay(model, input_tensor, rgb_img, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img / 255.0, grayscale_cam, use_rgb=True)
    return visualization

# target_layer for ResNet-50: model.layer4[-1]
```

Output this alongside the prediction in the UI — side by side with the original image.

---

## 7. Advisory Engine — The Anti-Hallucination Layer

### The core decision: NO free-form LLM generation for medical/treatment advice

This is the single most important design choice in the project, and you should state it explicitly to judges: **agricultural treatment advice is not something you let a generative model hallucinate**. A wrong disease treatment given confidently to a farmer is a real-world harm, not just a wrong chatbot answer.

**Design**: a static, hand-curated lookup table (JSON or Python dict) mapping each disease class to a fixed, pre-written recommendation. No LLM is involved in generating the *content* of the advice — only in translating pre-approved text to Kannada (Section 8, which is translation, not generation, and carries much lower hallucination risk).

```python
ADVISORY_DB = {
    "Tomato___Early_blight": {
        "cause": "Fungal infection (Alternaria solani), spreads in warm humid conditions.",
        "treatment": "Remove and destroy infected leaves. Apply a copper-based fungicide. Avoid overhead watering.",
        "prevention": "Rotate crops yearly. Ensure adequate spacing between plants for airflow."
    },
    "Maize___Common_rust": {
        "cause": "Fungal pathogen (Puccinia sorghi), spreads via airborne spores.",
        "treatment": "Apply fungicide at early symptom onset. Remove severely infected plants.",
        "prevention": "Plant rust-resistant maize varieties where available. Avoid dense planting."
    },
    # ... fill in for every class in your final dataset subset
}
```

**Where to source the actual agronomic content**: use established agricultural extension sources — ICAR (Indian Council of Agricultural Research), state agriculture department leaflets, or university agriculture extension pages — not an LLM's memory. Cross-check each entry against at least one authoritative source before adding it to the table. Keep a `sources.md` file listing where each entry's advice came from — this is exactly the kind of diligence that impresses judges in Q&A.

**If you're asked "why not use an LLM here?"** — your answer: deterministic, source-verified advice eliminates hallucination risk for a domain (crop treatment) where incorrect advice has real consequences for a farmer's livelihood. The LLM is used only downstream, for translation of already-verified text — a much lower-risk task.

---

## 8. Kannada Language Layer — Sarvam AI Integration

### Why Sarvam
Purpose-built for Indian languages, has a single SDK covering translation, speech-to-text, and text-to-speech across 22+ Indic languages including Kannada (`kn-IN`).

### SDK setup
```bash
pip install sarvamai --break-system-packages
```

```python
from sarvamai import SarvamAI
import os

client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
```

**Get an API key**: sign up at sarvam.ai, generate a subscription key, store it as an environment variable — never hardcode it in committed code (see Section 12 on secrets handling).

### Step 1: Translate advisory text to Kannada
```python
def translate_to_kannada(english_text: str) -> str:
    response = client.text.translate(
        input=english_text,
        source_language_code="en-IN",
        target_language_code="kn-IN"
    )
    return response.translated_text
```

### Step 2: Convert Kannada text to speech
```python
def kannada_text_to_speech(kannada_text: str, output_path="advisory_audio.wav"):
    response = client.text_to_speech.convert(
        text=kannada_text,
        target_language_code="kn-IN",
        model="bulbul:v3"
    )
    with open(output_path, "wb") as f:
        f.write(response.audio)  # confirm exact response field against current Sarvam docs when implementing
    return output_path
```

**Important**: verify the exact SDK method signatures against the live docs at https://docs.sarvam.ai when you implement this — SDKs change, and this guide reflects the API surface as of the time of writing. Do not assume the code above is 100% copy-paste correct; treat it as a strong starting scaffold.

### Optional stretch goal (only if core pipeline is done early): two-way voice Q&A
Farmer asks a follow-up question by voice → Saaras v3 (speech-to-text, `kn-IN`) transcribes it → feed the question + the diagnosis context into Sarvam's chat completion model (`sarvam-30b` or `sarvam-105b`) to generate a grounded answer → Bulbul speaks the response. This is a legitimate use of a generative model since it's answering a specific farmer question, not authoring the core medical advisory — but this is explicitly **future scope**, do not attempt before the static pipeline is solid.

---

## 9. Tech Stack Summary

### What we ARE using
- **PyTorch** — model training and inference
- **torchvision** — pretrained ResNet-50, image transforms
- **pytorch-grad-cam** — explainability layer
- **Sarvam AI SDK** — Kannada translation + text-to-speech
- **Streamlit** — UI wrapper (fastest path to a working demo; see Section 11)
- **scikit-learn** — evaluation metrics
- **Google Antigravity** — build environment (see Section 10)
- **Google Colab** — free GPU for training

### What we are explicitly NOT using, and why
- **No general-purpose LLM (GPT/Claude/Gemini) for generating treatment advice** — hallucination risk, see Section 7.
- **No live camera feed for the MVP** — descoped for time; static image upload only. Mention as future work.
- **No custom-collected/labeled dataset** — no time to gather and label real Karnataka field photos before the deadline; using PlantVillage is a stated, justified limitation, not a hidden one.
- **No mobile app** — a Streamlit web demo is sufficient for a stage presentation; a native app is future scope.

---

## 10. Building With Google Antigravity — Team Workflow

Antigravity is an agent-first IDE (VS Code-based, Gemini 3-powered, also supports Claude Sonnet as an alternate model) with two surfaces:
- **Editor view** — traditional hands-on coding with AI-assisted completions, good for focused work on one file/module.
- **Manager view** — lets you spawn and supervise multiple AI agents working on different tasks in parallel, each producing "Artifacts" (implementation plans, diffs, screenshots) you can review before accepting.

### How to use this guide inside Antigravity
Paste relevant sections of this document directly into an agent's task prompt. For example, for the training pipeline, give the agent Section 5 verbatim as its spec and ask it to implement `train.py` against it — this is exactly why this document is written in this much detail: it's meant to be a spec an agent can build against with minimal ambiguity.

### Recommended 3-person parallel split using Manager view
Each person runs their own Antigravity agent session, working from this shared document as the spec:

- **Person A — Model pipeline**: dataset filtering/loading (Section 4), model training (Section 5), evaluation (Section 5). Deliverable: a trained `.pth` weights file + evaluation report.
- **Person B — Explainability + Advisory**: Grad-CAM integration (Section 6), advisory lookup table with sourced content (Section 7). Deliverable: a module that takes an image + prediction and returns a heatmap + advisory text.
- **Person C — Kannada layer + UI**: Sarvam integration (Section 8), Streamlit UI (Section 11) that ties everything together. Deliverable: the user-facing app shell, initially working against dummy/placeholder model outputs so it doesn't block on Person A finishing training.

**Integration checkpoint**: schedule a merge point roughly halfway through your remaining time where Person C's UI shell gets wired to Person A's real model and Person B's real advisory output, replacing placeholders. Don't leave integration to the very end.

---

## 11. UI & Hosting

### UI: Streamlit (fastest path)
```bash
pip install streamlit --break-system-packages
```

```python
import streamlit as st
from PIL import Image

st.title("Sasya AI — ಸಸ್ಯ AI")
st.write("Upload a leaf photo to get an instant diagnosis in Kannada.")

uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded leaf")
    # -> run through: model → gradcam → advisory → sarvam translate → sarvam TTS
    # -> display prediction, confidence, heatmap image, Kannada text, and st.audio() for the spoken output
```

### Hosting options (both free, pick based on team comfort)
- **Streamlit Community Cloud** — connect your GitHub repo, deploys directly, simplest for a Streamlit app.
- **Hugging Face Spaces** — also free, supports Streamlit or Gradio, good if you want a slightly more ML-community-facing host.

For the actual stage presentation, **do not rely solely on live hosted deployment** — venue wifi is a real risk. Have the app running locally as backup, plus a recorded screen-capture video of a full successful run as a last-resort fallback.

---

## 12. Environment, Secrets, and Repo Structure

### Suggested repo structure
```
sasya-ai/
├── data/                      # filtered dataset (gitignored, too large for git)
├── models/
│   └── train.py
│   └── resnet50_sasya.pth     # trained weights (gitignored, share via Drive/Colab)
├── explainability/
│   └── gradcam.py
├── advisory/
│   └── advisory_db.json
│   └── sources.md
├── language/
│   └── sarvam_client.py
├── app.py                     # Streamlit entrypoint
├── requirements.txt
├── .env.example                # template, NOT actual keys
└── README.md
```

### Secrets handling
- Store `SARVAM_API_KEY` and any Kaggle credentials as environment variables or in a `.env` file that is in `.gitignore` — never commit real API keys to GitHub, even a private repo, and never paste them directly into a shared document.
- Provide a `.env.example` with placeholder values so teammates know what's needed without seeing real keys.

### requirements.txt (starting point)
```
torch
torchvision
grad-cam
sarvamai
streamlit
scikit-learn
pillow
numpy
```

---

## 13. Testing Checklist

- [ ] Model loads and runs inference on a single image without error
- [ ] Confusion matrix generated on held-out test set, reviewed for any class with very low recall (know why, be ready to explain)
- [ ] Grad-CAM overlay visibly highlights a plausible region on at least 5 sample images (not the whole image uniformly, not a random corner)
- [ ] Every class in your final dataset subset has a corresponding, sourced entry in `advisory_db.json` — no missing keys
- [ ] Sarvam translation output reviewed by a Kannada speaker on the team for correctness/naturalness before demo day
- [ ] Sarvam TTS audio actually plays correctly in the Streamlit app (`st.audio`) on the machine you'll demo from
- [ ] Full pipeline run end-to-end on at least 3 images the model has never seen (not from PlantVillage), to catch real-world failure modes before judges do
- [ ] App tested on venue-realistic conditions if possible (mobile hotspot instead of wifi, to simulate potential connectivity issues)
- [ ] Backup: a recorded video of one full successful run, saved locally, in case live demo fails

---

## 14. Anti-Hallucination Summary (for Q&A prep)

Be ready to explain each of these as deliberate choices:
1. **Classification is a trained CNN, not a language model** — it can be wrong, but it can't "make things up" the way a generative model can; its output space is a fixed, known set of classes with a confidence score.
2. **Explainability (Grad-CAM) lets a human sanity-check the model's reasoning** rather than trusting it blindly.
3. **Treatment advice comes from a curated, source-verified lookup table**, not generated text — zero hallucination surface for the highest-stakes content in the app.
4. **Kannada translation/TTS operates only on already-verified text** — translation carries far lower risk than generation, and any factual claim was fixed before translation touched it.
5. **Confidence thresholds**: consider adding a rule that if the model's confidence is below a set threshold (e.g. 60%), the app tells the farmer the result is uncertain and recommends consulting a local agricultural officer, rather than presenting a low-confidence guess as fact. This is a strong, easy-to-implement trust feature — mention it even if you only get to implement it partially.

---

## 15. What to explicitly call "future scope" in your pitch
State these openly rather than pretending the MVP is the finished product — judges respect clear scoping:
- Live camera feed / real-time detection
- Two-way voice conversation (farmer asks follow-up questions)
- Expanding beyond 4–5 crops to full regional coverage (ragi, areca nut, coconut, etc. — pending labeled data availability)
- Native mobile app for offline/low-connectivity rural use
- Integration with local agricultural extension officer contact/escalation for low-confidence cases

---

## Quick Start Command Summary

```bash
# 1. Clone base repo
git clone https://github.com/bPavan16/Plant-Disease-classification-using-CNN.git sasya-ai-base
cd sasya-ai-base

# 2. Set up environment
pip install torch torchvision grad-cam sarvamai streamlit scikit-learn pillow --break-system-packages

# 3. Get dataset
pip install kaggle --break-system-packages
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d data/

# 4. Set secrets
echo "SARVAM_API_KEY=your_key_here" > .env

# 5. Train (in Colab if no local GPU), then pull weights back
# 6. Run the app
streamlit run app.py
```

---

*This document is the shared spec for all three team members and for any AI coding agent (Antigravity, Claude, etc.) assisting on the build. Update it as decisions change — treat it as living documentation, not a one-time plan.*
