"""Generates notebooks/03_test_and_experiments.ipynb (course notebook 05c_Test_UploadedModel, plus the experiments)."""
from pathlib import Path

from nbtools import build, code, md

md("""
# Foto → Producto · Notebook 3: Test the uploaded model and experiments

Like the class notebook *05c_Test_UploadedModel*, the models are loaded **from Hugging Face**.
The project requires two kinds of experiments: (i) selecting the models (accuracy and runtime) and
(ii) the accuracy of the app on Streamlit Cloud.

| Step | Experiment | Measured |
|---|---|---|
| 1 | Load the testing data (962 photos the model never saw) | |
| 2 | **Model selection, classifier:** pre-trained model without fine-tuning vs our fine-tuned model | accuracy, runtime |
| 3 | **Model selection, description:** BLIP base vs BLIP large | runtime + human score |
| 4 | **The app's steps on real phone photos** | accuracy + human score |
| 5 | Everything in one Excel file, including the app's accuracy on Streamlit Cloud | |

**Runtime → Change runtime type → T4 GPU.** Accuracy is measured on the GPU (faster); the time per photo is
measured on the **CPU**, because Streamlit Cloud has no GPU.
""")

md("# Step 0: Installation of Packages")
code("""
!pip install "transformers<5.0.0" datasets sentencepiece sacremoses openpyxl -q
""")

md("""
# Configuration

The app's helper file (`product_card.py`), the real photos and notebook 1's results are taken from the project's
public GitHub repository, so this notebook tests exactly what the app uses.
""")
code("""
import os, sys, time, subprocess
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_dataset
from transformers import pipeline

MODEL_REPO = "boofalcon/foto-a-producto-vit"                    # our fine-tuned model
DATASET_REPO = "boofalcon/foto-a-producto-dataset"              # our balanced split (notebook 2)
BASELINE_MODEL = "google/siglip-base-patch16-224"               # pre-trained, no fine-tuning
CAPTION_MODELS = ["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"]
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-es"
GITHUB_REPO = "https://github.com/claudiafalcon/foto-a-producto.git"
GPU = 0 if torch.cuda.is_available() else -1

if not os.path.isdir("repo"):
    subprocess.run(["git", "clone", "-q", "--depth", "1", GITHUB_REPO, "repo"], check=True)
sys.path.insert(0, "repo")
from product_card import product_name, clean_caption

print("GPU:", torch.cuda.is_available())
""")

md("""
# Step 1: Load the testing data

The 962 testing photos from notebook 1, loaded from Hugging Face.
""")
code("""
def load_test_data():
    test = load_dataset(DATASET_REPO, split="test")
    images = [image.convert("RGB") for image in test["image"]]
    return images, list(test["category"])


def ms_per_photo(predict, images):
    \"\"\"Average milliseconds per photo, one photo at a time like in the app.\"\"\"
    predict(images[0])                                # the first call is slower (warm-up), not counted
    start = time.perf_counter()
    for image in images:
        predict(image)
    return round((time.perf_counter() - start) / len(images) * 1000)
""")

md("""
# Step 2: Model selection: the classifier

Is fine-tuning worth it? We compare our model with a strong pre-trained model **without fine-tuning**:
SigLIP, a *zero-shot* image classifier. It receives the category names as text and picks the one that best
matches the photo, so it needs no training. Both are tested on the same 962 photos.
SigLIP needs the categories in English, so each Spanish category gets an English description.
""")
code("""
ENGLISH_LABELS = {
    "a photo of a t-shirt": "Playera", "a photo of a shirt": "Camisa", "a photo of a women's top": "Blusa",
    "a photo of a dress": "Vestido", "a photo of jeans": "Jeans", "a photo of trousers": "Pantalón",
    "a photo of shorts": "Short", "a photo of socks": "Calcetines", "a photo of sneakers": "Tenis",
    "a photo of sandals": "Sandalias", "a photo of a handbag": "Bolsa", "a photo of a backpack": "Mochila",
    "a photo of a watch": "Reloj", "a photo of sunglasses": "Lentes de sol",
}


def zero_shot(device):
    classifier = pipeline("zero-shot-image-classification", model=BASELINE_MODEL, device=device)
    return lambda image: ENGLISH_LABELS[classifier(image, candidate_labels=list(ENGLISH_LABELS))[0]["label"]]


def fine_tuned(device):
    classifier = pipeline("image-classification", model=MODEL_REPO, device=device)
    return lambda image: classifier(image, top_k=1)[0]["label"]


def compare_classifiers(images, labels):
    rows = []
    for name, make in [(BASELINE_MODEL + " (zero-shot, no fine-tuning)", zero_shot),
                       (MODEL_REPO + " (fine-tuned)", fine_tuned)]:
        predict = make(GPU)
        accuracy = np.mean([predict(image) == label for image, label in zip(images, labels)])
        rows.append({"model": name, "test_photos": len(labels), "accuracy": round(accuracy, 4),
                     "cpu_ms_per_photo": ms_per_photo(make(-1), images[:30])})
    table = pd.DataFrame(rows)
    display(table)
    return table
""")

md("""
# Step 3: Model selection: the description

The image-to-text model writes an English sentence about the photo, which the app translates into Spanish.
We compare BLIP **base** and BLIP **large** on one testing photo per category. There is no automatic right answer
for a description, so the texts are scored by a person in the Excel file (`score_description`:
1 = usable as it is, 0.5 = needs a small edit, 0 = wrong), a subjective evaluation as allowed for generated text.
""")
code("""
def compare_captioners(images, labels):
    sample = pd.DataFrame({"category": labels}).groupby("category").head(1).index.tolist()   # one photo per category
    captions, speed = [], []
    for model_name in CAPTION_MODELS:
        captioner = pipeline("image-to-text", model=model_name, device=GPU)
        for i in sample:
            captions.append({"model": model_name, "category": labels[i],
                             "caption_en": clean_caption(captioner(images[i])[0]["generated_text"]),
                             "score_description": None})
        cpu_captioner = pipeline("image-to-text", model=model_name, device=-1)
        speed.append({"model": model_name,
                      "parameters_millions": round(sum(p.numel() for p in captioner.model.parameters()) / 1e6),
                      "cpu_ms_per_photo": ms_per_photo(cpu_captioner, [images[i] for i in sample])})
    captions, speed = pd.DataFrame(captions), pd.DataFrame(speed)
    display(captions.pivot(index="category", columns="model", values="caption_en"))
    display(speed)
    return captions, speed
""")

md("""
# Step 4: The app's steps on real phone photos

The same steps as the app (classify → describe → translate → name) on the photos the author took with a phone
(`eval_images/real_photos/<category>/`; the folder name is the true category). They are very different from the
studio photos used for training, so this is the closest test to a real merchant.
""")
code("""
def load_real_photos(folder="repo/eval_images/real_photos"):
    images, labels, files = [], [], []
    for category in sorted(os.listdir(folder)):
        for file in sorted(os.listdir(os.path.join(folder, category))):
            images.append(Image.open(os.path.join(folder, category, file)).convert("RGB"))
            labels.append(category)
            files.append(f"{category}/{file}")
    return images, labels, files


def run_app_steps(images, labels, files):
    classifier = pipeline("image-classification", model=MODEL_REPO, device=GPU)
    captioner = pipeline("image-to-text", model=CAPTION_MODELS[0], device=GPU)
    translator = pipeline("translation", model=TRANSLATION_MODEL, device=GPU)
    rows = []
    for image, true_category, file in zip(images, labels, files):
        top = classifier(image, top_k=1)[0]
        caption = clean_caption(captioner(image)[0]["generated_text"])
        rows.append({"photo": file, "true": true_category, "predicted": top["label"],
                     "correct": top["label"] == true_category, "confidence": round(top["score"], 3),
                     "name": product_name(top["label"], caption),
                     "description": translator(caption)[0]["translation_text"],
                     "score_name": None, "score_description": None})
    results = pd.DataFrame(rows)
    print(f"Accuracy on real photos: {results['correct'].mean():.1%} ({results['correct'].sum()} of {len(results)})")
    display(results)
    return results
""")

md("""
# Step 5: Excel file with all the results

One sheet per experiment. Notebook 1's tables and the app's accuracy on Streamlit Cloud (the CSV files downloaded
from the app's Evaluation tab) are read from the repository's `results/` folder.
""")
code("""
def export_excel(sheets, file="Experimental_results.xlsx"):
    for csv in sorted(os.listdir("repo/results")):
        sheets[csv.replace(".csv", "")] = pd.read_csv(os.path.join("repo/results", csv))
    with pd.ExcelWriter(file) as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    print("Sheets:", list(sheets))
    try:
        from google.colab import files
        files.download(file)
    except ImportError:
        print("Saved", file)
""")

md("# Run everything")
code("""
def main():
    print("Step 1: testing data")
    images, labels = load_test_data()

    print("Step 2: classifier, zero-shot vs fine-tuned")
    classifiers = compare_classifiers(images, labels)

    print("Step 3: description, BLIP base vs large")
    captions, caption_speed = compare_captioners(images, labels)

    print("Step 4: real phone photos")
    real = run_app_steps(*load_real_photos())

    print("Step 5: Excel")
    export_excel({"model_selection_classifier": classifiers, "model_selection_captioner": caption_speed,
                  "captions_to_score": captions, "real_photos": real})


if __name__ == "__main__":
    main()
""")

if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "notebooks" / "03_test_and_experiments.ipynb")
