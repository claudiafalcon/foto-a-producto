"""Builds eval_images/ for the app's Evaluation tab (run after notebook 1, from the project folder).

eval_images/dataset_test/<category>/  : N_PER_CATEGORY photos per category from the held-out test split
eval_images/real_photos/<category>/   : the author's own phone photos (data/real_photos), resized
"""
import os, shutil
import pandas as pd
from PIL import Image, ImageOps
try:
    from pillow_heif import register_heif_opener   # iPhone .HEIC photos
    register_heif_opener()
except ImportError:
    pass

N_PER_CATEGORY = 10
SEED = 42


def dataset_test_sample():
    test = pd.read_csv("data/test.csv")
    sample = test.groupby("category", group_keys=False).sample(n=N_PER_CATEGORY, random_state=SEED)
    for _, row in sample.iterrows():
        out = os.path.join("eval_images", "dataset_test", row["category"])
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join("data", row["file"]), os.path.join(out, os.path.basename(row["file"])))
    print("dataset_test:", len(sample), "photos")


def real_photos(max_side=640):
    count = 0
    for category in sorted(os.listdir("data/real_photos")):
        src = os.path.join("data/real_photos", category)
        if not os.path.isdir(src):
            continue
        for i, name in enumerate(sorted(f for f in os.listdir(src) if not f.startswith("."))):
            try:
                img = ImageOps.exif_transpose(Image.open(os.path.join(src, name))).convert("RGB")
            except Exception as e:
                print("skipped", category, name, e); continue
            img.thumbnail((max_side, max_side))
            out = os.path.join("eval_images", "real_photos", category)
            os.makedirs(out, exist_ok=True)
            img.save(os.path.join(out, f"{i + 1:02d}.jpg"), quality=90)
            count += 1
    print("real_photos:", count, "photos")


if __name__ == "__main__":
    shutil.rmtree("eval_images", ignore_errors=True)
    dataset_test_sample()
    real_photos()
