"""Create a stratified annotation sheet from valid image reviews."""
from pathlib import Path

import pandas as pd

SEED = 42
SAMPLE_SIZES = {"Negative": 250, "Neutral": 150, "Positive": 200}
INFO_COLS = ["review_id", "parent_asin", "split", "rating", "sentiment",
             "review_text", "image_path"]
LABEL_COLS = ["aspect", "emotion", "intent", "defect_type", "severity",
              "image_relevance", "text_image_consistency"]


def main() -> None:
    """Sample reviews per sentiment and write a CSV with empty label columns."""
    df = pd.read_csv("data/processed/image_reviews_valid.csv")
    parts = []
    for sentiment, n in SAMPLE_SIZES.items():
        subset = df[df["sentiment"] == sentiment]
        parts.append(subset.sample(min(n, len(subset)), random_state=SEED))
    sheet = pd.concat(parts).sample(frac=1, random_state=SEED)[INFO_COLS].copy()
    for col in LABEL_COLS:
        sheet[col] = ""

    out_dir = Path("data/annotation")
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out_dir / "annotation_sheet.csv", index=False,
                 encoding="utf-8-sig")
    print("Saved", len(sheet), "rows")
    print(sheet["split"].value_counts())


if __name__ == "__main__":
    main()
