"""Download review images, validate them, and record local paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

PROC = Path("data/processed")
IMG_DIR = Path("data/images")
MAX_SIDE = 512
MIN_SIDE = 64
TIMEOUT = 15
WORKERS = 16


def download_one(review_id: str, url: str):
    """Download one image; return (review_id, local_path or None, status)."""
    out = IMG_DIR / f"{review_id}.jpg"
    if out.exists():
        return review_id, str(out), "cached"
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        img.load()  # forces full decode, catches corrupt files
        img = img.convert("RGB")
        if min(img.size) < MIN_SIDE:
            return review_id, None, "too_small"
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img.save(out, "JPEG", quality=90)
        return review_id, str(out), "ok"
    except Exception as exc:  # network, decode, or format errors
        return review_id, None, type(exc).__name__


def main(limit: int | None) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC / "image_review_metadata.csv")
    if limit:
        df = df.head(limit)

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download_one, r, u)
                   for r, u in zip(df["review_id"], df["image_url"])]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            results.append(fut.result())

    res = pd.DataFrame(results, columns=["review_id", "image_path", "status"])
    print(res["status"].value_counts())

    df = df.merge(res, on="review_id")
    df = df[df["image_path"].notna()].copy()

    split_map = {}
    for name in ["train", "val", "test"]:
        ids = pd.read_csv(PROC / f"{name}.csv", usecols=["review_id"])
        split_map.update({i: name for i in ids["review_id"]})
    df["split"] = df["review_id"].map(split_map)

    df.to_csv(PROC / "image_reviews_valid.csv", index=False)
    print("Valid image reviews:", len(df))
    print(df["split"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    main(parser.parse_args().limit)