"""Foto → Producto: a photo of an item becomes a proposed catalog entry for a Nahui merchant.

Three Hugging Face pipelines:
  1. image-classification : fine-tuned ViT  -> category
  2. image-to-text        : BLIP            -> English description of the photo
  3. translation en -> es : Opus-MT         -> Spanish description
The merchant reviews and edits the proposal before adding it to the catalog.
"""
import os

import pandas as pd
import streamlit as st
from PIL import Image
from transformers import pipeline

from product_card import product_name, clean_caption

CLASSIFIER_MODEL = "boofalcon/foto-a-producto-vit"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-es"
LOW_CONFIDENCE = 0.50          # below this the app asks the merchant to check the category
EVAL_FOLDER = "eval_images"    # labelled photos for the Evaluation tab: eval_images/<set>/<category>/<photo>


@st.cache_resource(show_spinner="Cargando modelos… / Loading models…")
def load_pipelines():
    """Loads the three models once; every visitor reuses them."""
    classifier = pipeline("image-classification", model=CLASSIFIER_MODEL)
    captioner = pipeline("image-to-text", model=CAPTION_MODEL)
    translator = pipeline("translation", model=TRANSLATION_MODEL)
    return classifier, captioner, translator


def propose_entry(pipes, image):
    """One photo -> category (top 3 with scores), Spanish name and Spanish description."""
    classifier, captioner, translator = pipes
    top3 = classifier(image, top_k=3)
    category = top3[0]["label"]
    caption = clean_caption(captioner(image)[0]["generated_text"])
    description = translator(caption)[0]["translation_text"]
    return {"category": category, "confidence": top3[0]["score"], "top3": top3,
            "name": product_name(category, caption), "description": description, "caption_en": caption}


def photo_tab(pipes):
    categories = sorted(pipes[0].model.config.id2label.values())
    uploaded = st.file_uploader("Sube una foto / Upload a photo", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("No pude abrir el archivo. ¿Es una foto? / Could not open this file.")
        return

    if st.session_state.get("photo") != uploaded.file_id:     # analyse each photo only once
        with st.spinner("Analizando la foto… / Analysing…"):
            st.session_state.entry = propose_entry(pipes, image)
        st.session_state.photo = uploaded.file_id
    entry = st.session_state.entry

    left, right = st.columns([1, 2])
    left.image(image)
    with right:
        st.subheader(f"Categoría / Category: {entry['category']} ({entry['confidence']:.0%})")
        if entry["confidence"] < LOW_CONFIDENCE:
            st.warning("No estoy seguro de la categoría, revísala. / Low confidence: please check the category.")
        st.caption("Opciones / Options: " + ", ".join(f"{p['label']} ({p['score']:.0%})" for p in entry["top3"]))
        with st.form("entry"):
            category = st.selectbox("Categoría", categories, index=categories.index(entry["category"]))
            name = st.text_input("Nombre", entry["name"])
            description = st.text_area("Descripción", entry["description"])
            if st.form_submit_button("Agregar al catálogo / Add to catalog", type="primary"):
                st.session_state.catalog.append({"Categoría": category, "Nombre": name, "Descripción": description})
                st.success(f"Agregado / Added: {name}")

    st.subheader(f"Tu catálogo / Your catalog ({len(st.session_state.catalog)})")
    if st.session_state.catalog:
        table = pd.DataFrame(st.session_state.catalog)
        st.dataframe(table, hide_index=True)
        st.download_button("Descargar CSV / Download CSV", table.to_csv(index=False).encode("utf-8"),
                           "catalogo.csv", "text/csv")


def evaluation_tab(pipes):
    """Accuracy of the app as deployed: the category predicted for each labelled photo vs its folder name."""
    st.write("Runs the app on the labelled photos in `eval_images/` and compares the predicted category "
             "with the true one (the folder name).")
    sets = sorted(os.listdir(EVAL_FOLDER)) if os.path.isdir(EVAL_FOLDER) else []
    if not sets:
        st.info("No evaluation photos in this deployment.")
        return
    chosen = st.selectbox("Photo set", sets)
    if not st.button("Run evaluation"):
        return
    folder = os.path.join(EVAL_FOLDER, chosen)
    photos = [(category, file) for category in sorted(os.listdir(folder)) if not category.startswith(".")
              for file in sorted(os.listdir(os.path.join(folder, category))) if not file.startswith(".")]
    rows, progress = [], st.progress(0.0)
    for n, (true_category, file) in enumerate(photos, 1):
        image = Image.open(os.path.join(folder, true_category, file)).convert("RGB")
        entry = propose_entry(pipes, image)
        rows.append({"photo": f"{true_category}/{file}", "true": true_category, "predicted": entry["category"],
                     "correct": entry["category"] == true_category, "confidence": round(entry["confidence"], 3),
                     "name": entry["name"], "description": entry["description"]})
        progress.progress(n / len(photos))
    results = pd.DataFrame(rows)
    st.metric("Accuracy", f"{results['correct'].mean():.1%}", f"{results['correct'].sum()} / {len(results)} photos")
    st.dataframe(results, hide_index=True)
    st.download_button("Download results (CSV)", results.to_csv(index=False).encode("utf-8"),
                       f"app_evaluation_{chosen}.csv", "text/csv")


def main():
    st.set_page_config(page_title="Foto → Producto", page_icon="📸")
    st.title("📸 Foto → Producto")
    st.write("Toma una foto de un artículo y la app te propone la categoría, el nombre y la descripción "
             "para tu catálogo. Tú revisas y corriges antes de guardar.")
    with st.expander("About (English)"):
        st.write("ISOM5240 (HKUST) course project for Nahui (https://about.nahui.app). A fine-tuned ViT predicts "
                 "the category, BLIP describes the photo and Opus-MT translates the description into Spanish. "
                 "The merchant reviews everything before saving.")
        st.write(f"Models: `{CLASSIFIER_MODEL}` · `{CAPTION_MODEL}` · `{TRANSLATION_MODEL}`")

    if "catalog" not in st.session_state:
        st.session_state.catalog = []
    pipes = load_pipelines()

    photo, evaluation = st.tabs(["📷 Foto / Photo", "📊 Evaluación / Evaluation"])
    with photo:
        photo_tab(pipes)
    with evaluation:
        evaluation_tab(pipes)


if __name__ == "__main__":
    main()
