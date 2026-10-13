"""Photo → Product: a photo of an item becomes a proposed catalog entry for a Nahui merchant.

The merchant uploads a photo and the app:
    1. Classifies the item          -> classify()   image-classification, our fine-tuned ViT
    2. Describes the photo          -> describe()   image-to-text, BLIP
    3. Translates the description   -> translate()  translation English -> Spanish, Opus-MT
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
LOGO = "assets/nahui_logo.png"


# ---------- Models ----------

@st.cache_resource(show_spinner="Loading the models (the first time takes a few minutes)…")
def load_pipelines():
    """Loads the three models once; every visitor reuses them."""
    classifier = pipeline("image-classification", model=CLASSIFIER_MODEL)
    captioner = pipeline("image-to-text", model=CAPTION_MODEL)
    translator = pipeline("translation", model=TRANSLATION_MODEL)
    return classifier, captioner, translator


def classify(image, classifier):
    """Pipeline 1: the three most likely categories with their scores."""
    return classifier(image, top_k=3)


def describe(image, captioner):
    """Pipeline 2: an English sentence about the photo."""
    return clean_caption(captioner(image)[0]["generated_text"])


def translate(text, translator):
    """Pipeline 3: the English sentence in Spanish."""
    return translator(text)[0]["translation_text"]


def propose_entry(pipes, image, show_step=None):
    """Runs the three pipelines in order and builds the proposed catalog entry."""
    classifier, captioner, translator = pipes
    show_step = show_step or (lambda message: None)
    show_step("1 · Classifying the item…")
    top3 = classify(image, classifier)
    show_step("2 · Describing the photo…")
    caption = describe(image, captioner)
    show_step("3 · Translating into Spanish…")
    description = translate(caption, translator)
    category = top3[0]["label"]
    return {"category": category, "confidence": top3[0]["score"], "top3": top3, "caption_en": caption,
            "description": description, "name": product_name(category, caption)}


# ---------- Page ----------

def apply_styles():
    st.markdown("""
    <style>
      .block-container {max-width: 980px; padding-top: 2rem;}
      .hero {text-align: center; margin-bottom: 1.2rem;}
      .hero h1 {font-size: 2.6rem; margin-bottom: 0.2rem;}
      .hero p {color: #6B6259; font-size: 1.05rem;}
      .step {background: #ffffff; border: 1px solid #E8E2DC; border-radius: 16px; padding: 1rem 1.1rem;
             height: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.04);}
      .step .num {display: inline-block; width: 1.8rem; height: 1.8rem; border-radius: 50%; background: #C13F26;
                  color: #fff; text-align: center; line-height: 1.8rem; font-weight: 700; margin-right: 0.4rem;}
      .step .title {font-weight: 700; font-size: 1.05rem;}
      .step .model {color: #6B6259; font-size: 0.8rem; margin-top: 0.4rem;}
      .step .result {font-size: 1.15rem; margin-top: 0.6rem;}
      .step.done {border: 2px solid #C13F26;}
    </style>
    """, unsafe_allow_html=True)


def step_card(number, title, task, model, result=None):
    done = " done" if result is not None else ""
    body = f'<div class="result">{result}</div>' if result is not None else ""
    return (f'<div class="step{done}"><span class="num">{number}</span><span class="title">{title}</span>'
            f'<div class="model"><code>{task}</code> · {model}</div>{body}</div>')


def header():
    if os.path.exists(LOGO):
        st.image(LOGO, width=72)
    st.markdown('<div class="hero"><h1>📸 Photo → Product</h1>'
                '<p>Upload a photo of an item and get a catalog entry in Spanish, ready for the merchant to review.<br>'
                '<i>ISOM5240 (HKUST) course project proposal for <a href="https://about.nahui.app">Nahui</a></i></p></div>',
                unsafe_allow_html=True)


def show_pipeline(proposal=None):
    """The three steps as cards; after a photo is analysed each card shows its result."""
    if proposal:
        results = [f"<b>{proposal['category']}</b> ({proposal['confidence']:.0%})",
                   proposal["caption_en"], f"<b>{proposal['description']}</b>"]
    else:
        results = [None, None, None]
    steps = [("Classify", "image-classification", "fine-tuned ViT"),
             ("Describe", "image-to-text", "BLIP"),
             ("Translate", "translation en → es", "Opus-MT")]
    for i, (column, (title, task, model), result) in enumerate(zip(st.columns(3), steps, results), 1):
        column.markdown(step_card(i, title, task, model, result), unsafe_allow_html=True)


def photo_tab(pipes):
    categories = sorted(pipes[0].model.config.id2label.values())
    uploaded = st.file_uploader("Upload a photo of one item", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        show_pipeline()
        return
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("This file could not be opened. Is it a photo?")
        return

    if st.session_state.get("photo_id") != uploaded.file_id:     # analyse each photo only once
        with st.status("Running the three models…", expanded=True) as status:
            st.session_state.proposal = propose_entry(pipes, image, show_step=st.write)
            status.update(label="Done", state="complete", expanded=False)
        st.session_state.photo_id = uploaded.file_id
    proposal = st.session_state.proposal

    left, middle, right = st.columns([1, 2, 1])
    middle.image(image)
    show_pipeline(proposal)
    if proposal["confidence"] < LOW_CONFIDENCE:
        st.warning("Low confidence: please check the category. Other options: " +
                   ", ".join(f"{p['label']} ({p['score']:.0%})" for p in proposal["top3"][1:]))

    st.subheader("✏️ Review and add to the catalog")
    with st.form("review_form"):
        category = st.selectbox("Category", categories, index=categories.index(proposal["category"]))
        name = st.text_input("Name", proposal["name"])
        description = st.text_area("Description", proposal["description"])
        if st.form_submit_button("Add to catalog", type="primary"):
            st.session_state.catalog.append({"Categoría": category, "Nombre": name, "Descripción": description})
            st.success(f"Added: {name}")

    st.subheader(f"🗂️ Your catalog ({len(st.session_state.catalog)} items)")
    if st.session_state.catalog:
        table = pd.DataFrame(st.session_state.catalog)
        st.dataframe(table, hide_index=True)
        st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"), "catalogo.csv", "text/csv")


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
    st.set_page_config(page_title="Photo → Product", page_icon="📸", layout="centered")
    apply_styles()
    header()
    if "catalog" not in st.session_state:
        st.session_state.catalog = []
    pipes = load_pipelines()

    photo, evaluation = st.tabs(["📷 Photo", "📊 Evaluation"])
    with photo:
        photo_tab(pipes)
    with evaluation:
        evaluation_tab(pipes)


if __name__ == "__main__":
    main()
