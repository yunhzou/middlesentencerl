import argparse, json, uuid, multiprocessing as mp
from pathlib import Path

import numpy as np
import nltk, datasets
from datasets import load_from_disk
from nltk.tokenize import PunktSentenceTokenizer
from tqdm import tqdm

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

_tok = PunktSentenceTokenizer()
_MARKER = "\]"

def _next_non_ws(text: str, pos: int) -> int:
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos

def sent_starts_with_marker(text: str) -> np.ndarray:
    """Start indices using Punkt + custom '\]' terminator; last entry is len(text)."""
    starts = {0}

    # Punkt sentence starts
    for start, _ in _tok.span_tokenize(text):
        starts.add(start)

    # Custom '\]' sentence ends → next sentence starts after marker + following whitespace
    i = 0
    while True:
        j = text.find(_MARKER, i)
        if j == -1:
            break
        k = _next_non_ws(text, j + len(_MARKER))
        starts.add(k)
        i = j + 1  # allow overlapping scans safely

    starts = sorted(s for s in starts if 0 <= s <= len(text))
    if not starts or starts[-1] != len(text):
        starts.append(len(text))
    return np.asarray(starts, dtype=np.uint32)

def index_row(args):
    row, out_dir = args
    try:
        # Required columns
        answer = row["answer"]
        question = row.get("question")

        # Prefer a stable id if present; else UUID
        doc_id = (
            row.get("id")
            or row.get("doc_id")
            or row.get("doc", {}).get("arxiv_id")
            or str(uuid.uuid4())
        )

        # Write answer text and its index only (question is NOT indexed)
        txt_path = out_dir / f"{doc_id}.txt"
        txt_path.write_text(answer, encoding="utf-8")

        idx = sent_starts_with_marker(answer)
        (out_dir / f"{doc_id}.idx").write_bytes(idx.tobytes())

        return {
            "id": doc_id,
            "txt": txt_path.name,
            "idx": f"{doc_id}.idx",
            "n_sent": len(idx) - 1,
            "question": question,  # passthrough metadata
        }
    except Exception as e:
        return {"_error": str(e)}

def main(dataset_dir, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(dataset_dir)
    n_docs = len(ds)

    def task_iter():
        for i in range(n_docs):
            yield (ds[i], out_dir)

    procs = mp.cpu_count()
    chunksize = max(1, n_docs // (procs * 8))

    errors = 0
    with mp.Pool(processes=procs) as pool, open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as fp:
        for result in tqdm(pool.imap_unordered(index_row, task_iter(), chunksize=chunksize), total=n_docs):
            if result is None or "_error" in result:
                errors += 1
                continue
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")

    indexed = n_docs - errors
    print(f"Indexed {indexed} docs into {out_dir}/ (errors: {errors})")

if __name__ == "__main__":
    import os 
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()   
    storage_path = Path(os.getenv("STORAGE_PATH"))
    if not storage_path:
        raise ValueError("STORAGE_PATH environment variable is not set.")

    main(
        dataset_dir=storage_path / "MegaScience",
        out_dir=storage_path / "MegaScience_Index",
    )
