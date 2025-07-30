#!/usr/bin/env python
"""
build_char_index.py
-------------------
Download the first N papers from the Hugging Face dataset
  allenai/olmo‑mix‑1124   (ArXiv source)  :contentReference[oaicite:0]{index=0}
and write, for each paper:

  • <id>.txt   – raw UTF‑8 text
  • <id>.idx   – uint32 array of sentence‑start **character offsets**
  • manifest.jsonl – one row per paper with metadata used by the loader
"""

import argparse, json, uuid, multiprocessing as mp
from pathlib import Path

import numpy as np
import nltk, datasets

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# --------------------------------------------------------------------- #
from nltk.tokenize import PunktSentenceTokenizer
_tok = PunktSentenceTokenizer()

def sent_starts(text: str) -> np.ndarray:
    """
    Return absolute character offsets of every sentence start
    **plus a final sentinel len(text)**.
    """
    starts = [0]                                           # first sentence
    for start, _ in _tok.span_tokenize(text):
        if start != 0:                                     # span_tokenize includes 0
            starts.append(start)
    if starts[-1] != len(text):
        starts.append(len(text))                           # closing sentinel
    return np.asarray(starts, dtype=np.uint32)


def index_row(row, out_dir: Path):
    """Write .txt / .idx for one dataset row; return manifest entry."""
    text   = row["text"]
    doc_id = row.get("doc", {}).get("arxiv_id") or str(uuid.uuid4())

    # save raw text
    txt_path = out_dir / f"{doc_id}.txt"
    txt_path.write_text(text, encoding="utf-8")

    # save sentence‑offset index
    idx = sent_starts(text)
    (out_dir / f"{doc_id}.idx").write_bytes(idx.tobytes())

    return {"id": doc_id,
            "txt": txt_path.name,
            "idx": f"{doc_id}.idx",
            "n_sent": len(idx) - 1}


def main(out_dir: Path, n_docs: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    # stream = datasets.load_dataset("allenai/olmo-mix-1124",
    #                                split="train", streaming=True)
    stream = datasets.load_dataset("oscar","unshuffled_deduplicated_en",
                                split="train", streaming=True)
    rows = [row for _, row in zip(range(n_docs), stream)]

    with mp.Pool() as pool:
        manifest = pool.starmap(index_row,
                                [(row, out_dir) for row in rows])

    with open(out_dir / "manifest.jsonl", "w") as fp:
        for m in manifest:
            fp.write(json.dumps(m) + "\n")

    print(f"Indexed {len(manifest)} papers into {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=Path("olmo_char_demo"))
    p.add_argument("--n_docs", type=int, default=10,
                   help="number of papers to index (default 10)")
    main(**vars(p.parse_args()))
