"""Photo → Product: a photo of an item becomes a proposed catalog entry for a Nahui merchant.

The merchant uploads a photo and the app:
    1. Classifies the item          -> classify()   image-classification, our fine-tuned ViT
    2. Describes the photo          -> describe()   image-to-text, BLIP
    3. Translates the description   -> translate()  translation English -> Spanish, Opus-MT
The merchant reviews and edits the proposal before adding it to the catalog.
"""
import base64
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
    """Same look as the Nahui app: warm background, tinted product-tag cards, terracotta pill buttons."""
    st.markdown("""
    <style>
      .block-container {max-width: 900px; padding-top: 2rem;}
      .hero {display: flex; gap: 1.4rem; align-items: center; background: #F8EFEC; border-radius: 24px;
             padding: 1.6rem 1.8rem; margin-bottom: 1.4rem; border-bottom: 3px dashed #E8D6D0;}
      .hero .logo {width: 104px; height: 104px; background: #FFFFFF; border-radius: 26px; padding: 14px;
                    box-shadow: 0 4px 14px rgba(167,44,44,0.15);}
      .kicker {color: #6B6259; font-weight: 700; letter-spacing: 0.14em; font-size: 0.8rem;}
      .hero h1 {font-size: 3rem; font-weight: 800; line-height: 1.1;
                margin: 0.2rem 0 0.4rem 0; padding: 0; color: #2D2D2D;}
      .hero .arrow {color: #E86248;}
      .hero p {color: #6B6259; font-size: 1.02rem; margin: 0;}
      .hero a {color: #B5462F; font-weight: 600;}
      .dotted {border-top: 2px dashed #D5CFCA; margin: 1.2rem 0;}
      .step {border-radius: 18px; padding: 0.9rem 1rem 1rem 1rem; height: 100%; border: 2px solid;}
      .step-1 {background: #DBC3B6; border-color: #8F5A48;}
      .step-2 {background: #E9C1BC; border-color: #D0574D;}
      .step-3 {background: #E3CCB6; border-color: #B9845A;}
      .step .num {display: inline-block; background: #A72C2C; color: #fff; border-radius: 999px;
                  padding: 0 0.6rem; font-weight: 700; font-size: 0.85rem; margin-right: 0.4rem;}
      .step .title {font-weight: 700; font-size: 1.1rem; color: #2D2D2D;}
      .step .model {color: #6B6259; font-size: 0.8rem; margin-top: 0.3rem;}
      .step .result {font-size: 1.1rem; margin-top: 0.6rem; color: #2D2D2D;}
      button[kind^="primary"], button[kind^="secondary"], [data-testid="stBaseButton-primary"],
      [data-testid="stBaseButton-secondary"] {border-radius: 999px !important;}
      [data-testid="stFormSubmitButton"] button {width: 100%; padding: 0.7rem; font-size: 1.1rem; font-weight: 600;}
      [data-testid="stForm"] {background: #F8EFEC; border-radius: 18px; border-color: #E8D6D0;}
    </style>
    """, unsafe_allow_html=True)


def step_card(number, title, task, model, result=None):
    body = f'<div class="result">{result}</div>' if result is not None else ""
    return (f'<div class="step step-{number}"><span class="num">{number}</span><span class="title">{title}</span>'
            f'<div class="model">{task} · {model}</div>{body}</div>')


def header():
    """Banner like the top of the Nahui app: logo, small title, big title and a short explanation."""
    logo = ""
    if os.path.exists(LOGO):
        with open(LOGO, "rb") as file:
            logo = f'<img src="data:image/png;base64,{base64.b64encode(file.read()).decode()}" class="logo">'
    st.markdown(f"""
    <div class="hero">
      {logo}
      <div>
        <div class="kicker">NAHUI · ISOM5240 COURSE PROJECT</div>
        <h1>Photo <span class="arrow">→</span> Product</h1>
        <p>Upload a photo of an item and get a catalog entry in Spanish, ready for the merchant to review.
           A course project proposal for <a href="https://about.nahui.app">Nahui</a>.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


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

    left, middle, right = st.columns([1, 1, 1])
    middle.image(image, width=260)
    show_pipeline(proposal)
    if proposal["confidence"] < LOW_CONFIDENCE:
        st.warning("Low confidence: please check the category. Other options: " +
                   ", ".join(f"{p['label']} ({p['score']:.0%})" for p in proposal["top3"][1:]))

    st.markdown('<div class="dotted"></div>', unsafe_allow_html=True)
    st.subheader("✏️ Review and add to the catalog")
    with st.form("review_form"):
        category = st.selectbox("Category", categories, index=categories.index(proposal["category"]))
        name = st.text_input("Name", proposal["name"])
        description = st.text_area("Description", proposal["description"])
        if st.form_submit_button("Add to catalog", type="primary", width="stretch"):
            st.session_state.catalog.append({"Categoría": category, "Nombre": name, "Descripción": description})
            st.success(f"Added: {name}")

    st.markdown('<div class="dotted"></div>', unsafe_allow_html=True)
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
    if st.button("Run evaluation"):
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
        st.session_state.setdefault("evaluations", {})[chosen] = pd.DataFrame(rows)   # kept after the page reruns

    results = st.session_state.get("evaluations", {}).get(chosen)
    if results is None:
        return
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
