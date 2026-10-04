<div align="center">

# Multi-Modal Retinal Fundus Scan Analysis

**Multi-label eye disease detection from both fundus scans and patient demographics, with an AI-generated diagnostic report.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-000000?logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-OpenAI-1C3C3C?logo=langchain&logoColor=white)

<img src="ezgif-6-c5f60fb3c1.gif" alt="Web app demo" width="720">

</div>

## Overview

A single fundus image rarely tells the whole story — both eyes, and the patient's age and sex, all carry diagnostic signal. This project combines them in one model:

1. **Upload** the patient's age, sex, and left/right fundus scans in the web app.
2. **Predict** — a multimodal CNN returns a confidence score for 8 disease classes.
3. **Explain** — an LLM turns the strongest predictions into a short report: what the condition is, possible treatments, and next steps.

## Disease Classes

| Code | Class | Code | Class |
|:---:|---|:---:|---|
| N | Normal | A | Age-related Macular Degeneration |
| D | Diabetic Retinopathy | H | Hypertension |
| G | Glaucoma | M | Myopia |
| C | Cataract | O | Other abnormalities |

## How It Works

### 1. Preprocessing

Every scan is cleaned up before it reaches the model ([`utils.py`](utils.py)):

`Crop black border` → `Resize to 224×224` → `CLAHE contrast boost (clip 2.0, 8×8 tiles)` → `Median blur (3×3)`

### 2. Model

<p align="center">
  <img src="Model-Arch.png" alt="Model architecture" width="560">
</p>

- **Image branch** — a shared ImageNet-pretrained ResNet-101 (frozen, up to `conv4_block23`) extracts features from each eye, which are refined by an attention block and a Squeeze-and-Excitation block.
- **Late fusion** — left and right eye features are concatenated just before the decision layers.
- **Demographics** — age and sex pass through a 2048-unit dense layer and are multiplied with the image features, so patient context re-weights what the model looks at.
- **Classifier** — global average pooling feeds a Discriminative RBM (128 hidden units) that outputs a probability for each class.

The design builds on [Fundus-DeepNet](https://doi.org/10.1016/j.inffus.2023.102059) (Al-Fahdawi et al., 2024).

### 3. AI Diagnostic Report

The patient details and predictions are sent to an OpenAI chat model through LangChain. The model is prompted to act as an ophthalmologist, focus on classes above 50% confidence (or the top one if none are), ignore *Other*, tailor advice to the patient's age and sex, and return:

**Disease description · Possible cures · Next steps**

## Dataset & Training

- **Data** — [OIA-ODIR](https://drive.google.com/file/d/1-7DO1jJFC_4W0hc2CaonlLe595M4eDOh/view): 5,000 patients with left/right fundus photos, age, sex, and doctor-labelled diagnoses, collected from hospitals across China.
- **Class balancing** — under-represented classes were augmented with Albumentations (rotation, flips, brightness/contrast, hue/saturation), growing the training set to ~18k samples.
- **Training** — Adam (lr `6e-4`), binary cross-entropy, 30 epochs, batch size 52.

The full pipeline — data prep, augmentation, model definition, training, and evaluation — is in [`FUNDUS-DEEP-NET-AUGMENTED.ipynb`](FUNDUS-DEEP-NET-AUGMENTED.ipynb).

## Results

| Split | Accuracy | AUC |
|---|:---:|:---:|
| Training | 91% | 1.00 |
| Test | 70% | 0.83 |

More detail in the [project presentation (PDF)](A%20Multimodal%20Approach%20for%20Ophthalmic%20Disease%20Diagnosis.pdf).

## Getting Started

**You'll need:** Python 3.11, an OpenAI API key, and a trained model saved from the notebook.

```bash
# Clone
git clone https://github.com/ImaadHasan2002/Multi-Modal-Retinal-Fundus-Scan-Analysis.git
cd Multi-Modal-Retinal-Fundus-Scan-Analysis

# Install dependencies (the model is a TF SavedModel, which Keras 3 can't load directly)
pip install "tensorflow<2.16" flask opencv-python pillow numpy langchain-core langchain-openai python-dotenv

# Add your OpenAI key
echo "OPENAI_API_KEY=your-key-here" > .env
```

Then point `load_model(...)` in [`utils.py`](utils.py) at your saved model folder and start the app:

```bash
python app.py
```

Open **http://localhost:3000**, enter the patient details, upload both scans, and click **Analyze**.

## API

| Method | Route | Input | Returns |
|---|---|---|---|
| `GET` | `/` | — | Web interface |
| `POST` | `/analyze` | Form data: `age`, `sex`, `left-scan`, `right-scan` | Confidence per class (JSON) |
| `POST` | `/cgpt_suggest` | — *(uses the last `/analyze` result)* | `{ "cgpt_text": "..." }` |

## Project Structure

```
├── app.py                            # Flask server: routes + LLM report
├── utils.py                          # Preprocessing + model inference
├── FUNDUS-DEEP-NET-AUGMENTED.ipynb   # Data prep, augmentation, training
├── templates/inference.html          # Web UI
├── static/                           # CSS + JavaScript
├── Model-Arch.png                    # Architecture diagram
└── A Multimodal Approach for Ophthalmic Disease Diagnosis.pdf
```

## Authors

**[Imaad Hasan](https://github.com/ImaadHasan2002)** · **[Faisal Ali Khan](https://github.com/FaisalKhan19)**
B.Tech Artificial Intelligence, Minor Project II · Supervised by **Dr. Junaid Ali Reshi**

## References

1. Al-Fahdawi, S. et al. (2024). *Fundus-DeepNet: Multi-label deep learning classification system for enhanced detection of multiple ocular diseases through data fusion of fundus images.* Information Fusion, 102, 102059. [doi:10.1016/j.inffus.2023.102059](https://doi.org/10.1016/j.inffus.2023.102059)
2. Li, N. et al. (2021). *A benchmark of ocular disease intelligent recognition: one-shot for multi-disease detection.* LNCS, 177–193. [doi:10.1007/978-3-030-71058-3_11](https://doi.org/10.1007/978-3-030-71058-3_11)

> [!WARNING]
> This is an academic research project, not a medical device. Its predictions and AI-generated reports must not be used for real diagnosis; always consult a qualified ophthalmologist.
