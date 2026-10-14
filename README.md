# Foto → Producto

ISOM5240 (HKUST) course project for **Nahui** — https://about.nahui.app

A Streamlit app that turns a photo of an item into a proposed catalog entry in Spanish
(category, name, description). The merchant reviews and edits the proposal before adding it to the catalog.

| | |
|---|---|
| App (Streamlit Community Cloud) | _URL after deployment_ |
| Fine-tuned model | https://huggingface.co/boofalcon/foto-a-producto-vit |
| Prepared dataset | https://huggingface.co/datasets/boofalcon/foto-a-producto-dataset |

## How it works

```
photo ──► image-classification  (fine-tuned google/vit-base-patch16-224)  ──► category
      └─► image-to-text         (Salesforce/blip-image-captioning-base)   ──► English description
                                                                              │
          product_card.py: keep the phrase about the predicted item,          ▼
          take its colour for the name  ──► translation (Helsinki-NLP/opus-mt-en-es) ──► Spanish description
```

## Repository

| Path | Purpose |
|---|---|
| `app.py` | Streamlit app (three Hugging Face pipelines) |
| `product_card.py` | Builds the Spanish name and description; shared by the app and notebook 3 |
| `requirements.txt` | Dependencies for Streamlit Community Cloud (CPU-only PyTorch) |
| `notebooks/01_finetuning.ipynb` | Dataset preparation, fine-tuning experiments, saving the model |
| `notebooks/02_upload_to_huggingface.ipynb` | Publishing the model and dataset |
| `notebooks/03_test_and_experiments.ipynb` | Baseline comparison, captioner choice, full-pipeline and real-photo tests, Excel export |
| `results/` | Tables produced by notebook 1 |
| `eval_images/` | Labelled photos used by the app's Evaluation tab |

## Data

[`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small),
a copy of the Kaggle dataset *Fashion Product Images (Small)* by Param Aggarwal (MIT licence).
14 categories × 460 photos, split 70 / 15 / 15 by product so that no product appears in two splits.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
