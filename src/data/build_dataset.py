"""Build a reproducible, cleaned, split subset of Amazon Reviews 2023."""
import gzip
import json
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW_PATH = Path("data/raw/Cell_Phones_and_Accessories.jsonl.gz")
OUT_DIR = Path("data/processed")
SEED = 42
MAX_IMAGE_REVIEWS = 20_000
MAX_TEXT_ONLY = 20_000
MAX_LINES = 3_000_000
MIN_WORDS = 3


def load_reviews(path: Path = RAW_PATH) -> pd.DataFrame:
    """Stream the raw file, keeping all image reviews up to a cap
    plus a random sample of text-only reviews."""
    rng = random.Random(SEED)
    rows, n_img, n_txt = [], 0, 0

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_LINES or (n_img >= MAX_IMAGE_REVIEWS
                                  and n_txt >= MAX_TEXT_ONLY):
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            images = r.get("images") or []
            has_image = len(images) > 0
            if has_image and n_img >= MAX_IMAGE_REVIEWS:
                continue
            if not has_image and (n_txt >= MAX_TEXT_ONLY
                                  or rng.random() > 0.05):
                continue

            rows.append({
                "user_id": r.get("user_id"),
                "parent_asin": r.get("parent_asin"),
                "asin": r.get("asin"),
                "rating": r.get("rating"),
                "title": r.get("title"),
                "text": r.get("text"),
                "timestamp": r.get("timestamp"),
                "verified_purchase": r.get("verified_purchase"),
                "n_images": len(images),
                "image_url": images[0].get("large_image_url") if has_image else None,
            })
            if has_image:
                n_img += 1
            else:
                n_txt += 1

    return pd.DataFrame(rows)


def clean_text(text: str) -> str:
    """Remove HTML tags/URLs and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_sentiment(rating: float) -> str:
    """Weak sentiment label derived from star rating."""
    if rating <= 2:
        return "Negative"
    return "Neutral" if rating == 3 else "Positive"


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Combine text, drop duplicates/short/invalid rows, add labels and IDs."""
    df = df.dropna(subset=["rating", "parent_asin"]).copy()
    df["review_text"] = (df["title"].fillna("") + ". "
                         + df["text"].fillna("")).map(clean_text)
    df = df[df["review_text"].str.split().str.len() >= MIN_WORDS]
    df = df.drop_duplicates(subset=["user_id", "parent_asin", "review_text"])
    df = df.drop_duplicates(subset=["parent_asin", "review_text"])
    df["sentiment"] = df["rating"].map(get_sentiment)
    df = df.reset_index(drop=True)
    df.insert(0, "review_id", [f"R{i:07d}" for i in range(len(df))])
    return df


def split_by_product(df: pd.DataFrame):
    """70/15/15 split grouped by product to avoid leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_idx, rest_idx = next(gss.split(df, groups=df["parent_asin"]))
    train, rest = df.iloc[train_idx], df.iloc[rest_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(gss2.split(rest, groups=rest["parent_asin"]))
    return train, rest.iloc[val_idx], rest.iloc[test_idx]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_reviews(load_reviews())
    print("Clean dataset:", df.shape)
    print("With images:", df["image_url"].notna().sum())
    print(df["sentiment"].value_counts(normalize=True).round(3))

    train, val, test = split_by_product(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(name, part.shape)

    df[df["image_url"].notna()].to_csv(
        OUT_DIR / "image_review_metadata.csv", index=False)


if __name__ == "__main__":
    main()