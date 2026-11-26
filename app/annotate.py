"""Streamlit tool for labeling the annotation sheet."""
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

SHEET = Path("data/annotation/annotation_sheet.csv")
OPTIONS = {
    "aspect": ["battery", "display", "camera", "design", "fit", "durability",
               "packaging", "charging", "quality", "price", "delivery",
               "other"],
    "emotion": ["satisfaction", "excitement", "frustration", "disappointment",
                "anger", "confusion", "neutral"],
    "intent": ["complaint", "praise", "inquiry", "refund", "replacement",
               "warranty", "other"],
    "defect_type": ["none", "crack", "scratch", "broken", "discoloration",
                    "peeling", "melting", "stitching", "packaging_damage",
                    "other"],
    "severity": ["none", "low", "medium", "high", "critical"],
    "image_relevance": ["yes", "no"],
    "text_image_consistency": ["supported", "contradicted", "unclear"],
}

st.set_page_config(page_title="Annotation tool", layout="wide")


def load() -> pd.DataFrame:
    """Read the sheet with every column as text."""
    return pd.read_csv(SHEET, dtype=str, keep_default_na=False,
                       encoding="utf-8-sig")


df = load()
done = int((df["aspect"] != "").sum())
st.title("Review annotation")
st.progress(done / len(df), text=f"{done} of {len(df)} labeled")

if "idx" not in st.session_state:
    unlabeled = df.index[df["aspect"] == ""]
    st.session_state.idx = int(unlabeled[0]) if len(unlabeled) else 0

idx = st.number_input("Row number", 0, len(df) - 1, st.session_state.idx)
st.session_state.idx = int(idx)
row = df.iloc[st.session_state.idx]

left, right = st.columns(2)
with left:
    if Path(row["image_path"]).is_file():
        st.image(Image.open(row["image_path"]), use_container_width=True)
    else:
        st.warning("Image file not found")
with right:
    st.write(f"**Rating:** {row['rating']}  |  **Sentiment:** "
             f"{row['sentiment']}")
    st.write(row["review_text"])

st.divider()
choices = {}
cols = st.columns(4)
for i, (name, opts) in enumerate(OPTIONS.items()):
    current = row[name]
    pos = opts.index(current) if current in opts else None
    choices[name] = cols[i % 4].selectbox(name, opts, index=pos,
                                          key=f"{name}_{st.session_state.idx}")

if st.button("Save and next"):
    if any(v is None for v in choices.values()):
        st.error("Please fill every field before saving.")
    else:
        for name, value in choices.items():
            df.loc[st.session_state.idx, name] = value
        df.to_csv(SHEET, index=False, encoding="utf-8-sig")
        st.session_state.idx = min(st.session_state.idx + 1, len(df) - 1)
        st.rerun()