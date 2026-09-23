# Contributing to Sasya AI

Welcome to the Sasya AI project! Since we are working against a hard deadline for the Bengaluru Tech Summit (Sep 25, 2026), we need to move fast without stepping on each other's toes. Please read this guide before pushing any code.

## Quickstart (If you just cloned this repo)
1. **Install Requirements:** 
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables:** 
   Copy `.env.example` to `.env` and fill in your real API keys (e.g., `SARVAM_API_KEY`). Do **not** commit the `.env` file!
3. **Get the Trained Weights:** 
   The trained PyTorch model (`resnet50_sasya.pth`) is too large for Git. Download it from our shared Google Drive (or whoever just finished a training run) and place it exactly at:
   `models/resnet50_sasya.pth`
   *(Without this, the app will run with a dummy prediction fallback.)*

---

## 1. Branch Strategy
- **Never commit directly to `main`.**
- Each person works on their own feature branch.
  - Examples: `person1-training`, `person2-dataset-consolidation`, `person3-advisory-language`
- Once your feature is working locally, push your branch and open a Pull Request (PR).

## 2. Separation of Concerns (Avoiding Conflicts)
Our repo structure separates concerns by folder. You should primarily touch your own folder:
- `models/` = Training and data prep scripts
- `advisory/` = Advisory engine and DB
- `language/` = Sarvam API integration
- `explainability/` = Grad-CAM logic
- `app.py` = The UI glue

**⚠️ CAUTION: Shared Files**
The following files are shared and highly prone to merge conflicts:
- `app.py` (imports from everyone's modules)
- `requirements.txt`
- `README.md`
If you need to edit these, either message the team first or keep your edit strictly minimal and additive (e.g., appending an import, not restructuring the whole file).

## 3. Pull-Before-Push Discipline
Always pull the latest changes and rebase *before* you push to avoid messy merge commits and catch conflicts locally:
```bash
git pull --rebase origin main
git push origin your-branch-name
```

## 4. Commit Message Convention
Keep them short and prefix them with the area of the repo you are touching.
- `[training] add baseline eval metrics`
- `[advisory] source 6 new classes`
- `[ui] update confidence display`

## 5. Pull Request (PR) Process
Even with just 3 people, open a PR into `main` rather than pushing directly. Have at least one other person glance at the diff before merging. This catches path mismatches and broken imports before they break the build for someone else.

## 6. What NOT to Commit
Our `.gitignore` is already set up to catch these, but be explicitly aware:
- **Trained Weights:** Never commit `.pth` files. They bloat the git history.
- **Data Folders:** Do not commit the `data/` folder containing images.
- **API Keys:** Never commit `.env` files.

---

## Current Task Split & Parallel Workstreams

### Person 1 — Baseline Training (Kaggle)
- Focuses entirely on running the baseline Kaggle Notebook (`baseline_training.ipynb`) using Kaggle's free GPU.
- Scope is strictly limited to the base Kaggle PlantVillage dataset, filtered to Maize, Tomato, Grape, and Potato.
- Generates the initial `resnet50_sasya_baseline.pth` weights and shares them with the team via Google Drive.

### Person 2 — Multi-dataset Consolidation (No GPU needed yet)
- Identify and shortlist 2-3 additional candidate datasets (e.g., PlantDoc, FieldPlant) that best match our Karnataka crop scope. Prioritize quality and consistent labeling over sheer quantity.
- Build a canonical class taxonomy (e.g., `Tomato___Early_Blight`, `Maize___Common_Rust`).
- Write a mapping script (extend `models/data_prep.py`) translating each additional dataset's folder/label names into this canonical naming.
- Resize/standardize images to 224x224 during consolidation, and tag each image with its source dataset (for later per-source accuracy checks).
- **Output:** A clean, merged train/valid folder structure ready to be dropped into a second training run.
- **CRITICAL:** Do NOT start a second training run until Person 1's baseline weights are downloaded and confirmed working in `app.py`.

### Person 3 — Explainability, Advisory Sourcing, and Language
- **Advisory Sources:** Finish sourcing real citations for `advisory/sources.md` (replace placeholder brackets with actual ICAR/state agricultural extension URLs). Flag any class you cannot find a solid source for.
- **Advisory DB Expansion:** Once Person 2 confirms the final merged class list, expand `advisory_db.json` to cover all of it, sourced properly.
- **Sarvam Fixes:** Verify and finalize the Sarvam integration fixes (handle `response.audios[0]` decoding and add `speaker` parameter) flagged in the last session. Test the full Translate + TTS flow with the real API key.
- **Confidence Threshold:** Implement the fallback described in the Build Guide (Section 14, point 5): If the model prediction is low-confidence, the UI should explicitly tell the farmer to consult a local agricultural officer rather than presenting an uncertain guess as fact.
