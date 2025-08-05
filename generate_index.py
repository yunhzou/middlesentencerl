#!/usr/bin/env python
import argparse, json, uuid, multiprocessing as mp
from pathlib import Path

import numpy as np
import nltk, datasets
from datasets import load_from_disk
from nltk.tokenize import PunktSentenceTokenizer
from tqdm import tqdm

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

_tok = PunktSentenceTokenizer()

def sent_starts(text: str) -> np.ndarray:
    starts = [0]
    for start, _ in _tok.span_tokenize(text):
        if start != 0:
            starts.append(start)
    if starts[-1] != len(text):
        starts.append(len(text))
    return np.asarray(starts, dtype=np.uint32)

def index_row(args):
    row, out_dir = args
    try:
        text   = row["rewritten_context"]
        doc_id = row.get("doc", {}).get("arxiv_id") or str(uuid.uuid4())

        txt_path = out_dir / f"{doc_id}.txt"
        txt_path.write_text(text, encoding="utf-8")

        idx = sent_starts(text)
        (out_dir / f"{doc_id}.idx").write_bytes(idx.tobytes())

        return {"id": doc_id, "txt": txt_path.name, "idx": f"{doc_id}.idx", "n_sent": len(idx) - 1}
    except Exception as e:
        # Return a marker so the main loop can skip / optionally log
        return {"_error": str(e)}

def main(dataset_dir, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(dataset_dir)
    n_docs = len(ds)

    # Generator of tasks to avoid materializing rows in memory
    def task_iter():
        for i in range(n_docs):
            yield (ds[i], out_dir)

    # A reasonable chunksize for mixed CPU/IO; adjust if needed
    procs = mp.cpu_count()
    chunksize = max(1, n_docs // (procs * 8))  # small batches to keep progress smooth

    errors = 0
    with mp.Pool(processes=procs) as pool, open(out_dir / "manifest.jsonl", "w") as fp:
        for result in tqdm(pool.imap_unordered(index_row, task_iter(), chunksize=chunksize), total=n_docs):
            if result is None or "_error" in result:
                errors += 1
                continue
            fp.write(json.dumps(result) + "\n")

    indexed = n_docs - errors
    print(f"Indexed {indexed} papers into {out_dir}/ (errors: {errors})")

if __name__ == "__main__":
    import os 
    from pathlib import Path
    storage_path = Path(os.getenv("STORAGE_PATH"))
    if not storage_path:
        raise ValueError("STORAGE_PATH environment variable is not set.")

    main(
        dataset_dir=storage_path / "AoPS-Instruct-merged_context",
        out_dir=storage_path / "AoPS-Instruct_Index",
    )
