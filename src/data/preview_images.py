"""Save a contact sheet of random review images with their text."""
from pathlib import Path

import pandas as pd
from PIL import Image

df = pd.read_csv("data/processed/image_reviews_valid.csv")
sample = df[df["sentiment"] == "Negative"].sample(12, random_state=1)

sheet = Image.new("RGB", (4 * 256, 3 * 256), "white")
for i, path in enumerate(sample["image_path"]):
    img = Image.open(path).resize((256, 256))
    sheet.paste(img, ((i % 4) * 256, (i // 4) * 256))

Path("data/processed").mkdir(exist_ok=True)
sheet.save("data/processed/preview.jpg")
for i, t in enumerate(sample["review_text"]):
    print(i, t[:120])
