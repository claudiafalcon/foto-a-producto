"""Turns the model outputs into a Spanish product entry (used by app.py and notebook 3)."""
import re

# Spanish colour words in four forms: masculine singular, feminine singular, masculine plural, feminine plural.
COLOURS = {
    "black": ("negro", "negra", "negros", "negras"),
    "white": ("blanco", "blanca", "blancos", "blancas"),
    "red": ("rojo", "roja", "rojos", "rojas"),
    "blue": ("azul", "azul", "azules", "azules"),
    "green": ("verde", "verde", "verdes", "verdes"),
    "yellow": ("amarillo", "amarilla", "amarillos", "amarillas"),
    "pink": ("rosa", "rosa", "rosa", "rosa"),
    "purple": ("morado", "morada", "morados", "moradas"),
    "grey": ("gris", "gris", "grises", "grises"),
    "gray": ("gris", "gris", "grises", "grises"),
    "brown": ("café", "café", "café", "café"),
    "orange": ("naranja", "naranja", "naranja", "naranja"),
    "silver": ("plateado", "plateada", "plateados", "plateadas"),
    "gold": ("dorado", "dorada", "dorados", "doradas"),
}

# Which colour form each category needs: (0 masculine / 1 feminine) + (0 singular / 2 plural).
COLOUR_FORM = {
    "Blusa": 1, "Bolsa": 1, "Camisa": 1, "Mochila": 1, "Playera": 1,
    "Pantalón": 0, "Reloj": 0, "Short": 0, "Vestido": 0,
    "Sandalias": 3,
    "Calcetines": 2, "Jeans": 2, "Lentes de sol": 2, "Tenis": 2,
}


def product_name(category, caption):
    """Category plus the first colour in the caption: 'Mochila roja'. Only the category if no colour is found."""
    for word in re.findall(r"[a-z]+", caption.lower()):
        if word in COLOURS:
            return f"{category} {COLOURS[word][COLOUR_FORM[category]]}"
    return category


def clean_caption(caption):
    """Removes the words that are about the photo, not the item: 'a photo of', and BLIP's 'arafed'."""
    text = re.sub(r"\b(arafed|araffe)\b", "", caption, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(a |an )?(photo|picture|close up) of ", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()
