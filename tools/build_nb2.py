"""Builds notebooks/02_upload_to_huggingface.ipynb (mirrors the course's 05b_Upload_SavedModel notebook)."""
from pathlib import Path
from nbtools import md, code, build, reset

reset()

md("""
# Foto → Producto · Notebook 2: upload the fine-tuned model to Hugging Face

Mirrors the course notebook *05b_Upload_SavedModel*. Run it **after** notebook 1.

| Step | What happens |
|---|---|
| 1 | Bring the saved model (`fine_tuned_model.zip`) from Google Drive into Colab |
| 2 | Create the model and image-processor objects from the saved folder |
| 3 | Log in to Hugging Face and push the model, the processor and the prepared dataset |
| 4 | Check: download the model back from Hugging Face and classify a few photos |

You need a Hugging Face **write** token (huggingface.co → Settings → Access Tokens).
This notebook does not need a GPU.
""")

md("# Step 0: Installation of packages")
code("""
!pip install "transformers<5.0.0" datasets -q
""")

md("""
# Step 1: Upload the saved model to Colab

Notebook 1 copied `fine_tuned_model.zip` (and `dataset_files.zip`) to `MyDrive/Models/`.
""")
code("""
from google.colab import drive
drive.mount('/content/drive')
""")
code("""
!unzip -o -q /content/drive/MyDrive/Models/fine_tuned_model.zip -d fine_tuned_model
!unzip -o -q /content/drive/MyDrive/Models/dataset_files.zip -d data
!ls fine_tuned_model data
""")

md("# Step 2: Create the model and image-processor objects")
code("""
from transformers import AutoModelForImageClassification, AutoImageProcessor

model = AutoModelForImageClassification.from_pretrained("fine_tuned_model")
processor = AutoImageProcessor.from_pretrained("fine_tuned_model")

print("Categories:", list(model.config.id2label.values()))
""")

md("""
# Step 3: Upload to Hugging Face

Paste your **write** token when asked (answer `n` to "Add token as git credential?").
""")
code("""
!hf auth login
""")
code("""
!hf auth whoami
""")
code("""
MODEL_REPO = "boofalcon/foto-a-producto-vit"          # <Hugging Face account>/<model name>
DATASET_REPO = "boofalcon/foto-a-producto-dataset"

model.push_to_hub(MODEL_REPO)
processor.push_to_hub(MODEL_REPO)          # the app needs the processor together with the model
print("Model:", "https://huggingface.co/" + MODEL_REPO)
""")

md("""
Also publish the prepared dataset (the exact train / validation / test split used for fine-tuning),
so anyone can check its size and class balance in the Hugging Face dataset viewer.
""")
code("""
import pandas as pd
from datasets import load_dataset, DatasetDict, Image

def load_split(split):
    \"\"\"The exported CSV of one split, with its image files attached.\"\"\"
    df = pd.read_csv(f"data/{split}.csv")
    df["image"] = "data/" + df["file"]
    ds = load_dataset("csv", data_files={split: f"data/{split}.csv"})[split]
    ds = ds.add_column("image", df["image"].tolist()).cast_column("image", Image())
    return ds.remove_columns(["file"])

prepared = DatasetDict({s: load_split(s) for s in ["train", "validation", "test"]})
print(prepared)
prepared.push_to_hub(DATASET_REPO)
print("Dataset:", "https://huggingface.co/datasets/" + DATASET_REPO)
""")

md("""
# Step 4: Check the uploaded model

Load the model **from Hugging Face** (not from the local folder) with a pipeline, exactly as the app will.
""")
code("""
from transformers import pipeline
from PIL import Image as PILImage

classifier = pipeline("image-classification", model=MODEL_REPO)
test = pd.read_csv("data/test.csv").sample(8, random_state=1)
for _, row in test.iterrows():
    top = classifier(PILImage.open("data/" + row["file"]).convert("RGB"), top_k=1)[0]
    print(f"true: {row['category']:<14} predicted: {top['label']:<14} score: {top['score']:.2f}")
""")

if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "notebooks" / "02_upload_to_huggingface.ipynb", gpu=False)
