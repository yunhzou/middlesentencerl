import json, uuid, multiprocessing as mp
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

def line_starts(text: str) -> np.ndarray:
    """Return start indices of each line in the text (split by \\n)."""
    starts = [0]
    offset = 0
    for line in text.splitlines(True):  # keep line endings to track offsets
        offset += len(line)
        starts.append(offset)
    if starts[-1] != len(text):
        starts.append(len(text))
    return np.asarray(starts, dtype=np.uint32)

def index_row(args):
    row, out_dir = args
    try:
        solution_text = row["solution"]   # main text
        input_text    = row["input"]
        doc_id = str(uuid.uuid4())

        # write solution text into a .txt file
        txt_path = out_dir / f"{doc_id}.txt"
        txt_path.write_text(solution_text, encoding="utf-8")

        # write line index for solution
        idx = line_starts(solution_text)
        (out_dir / f"{doc_id}.idx").write_bytes(idx.tobytes())

        return {
            "id": doc_id,
            "txt": txt_path.name,
            "idx": f"{doc_id}.idx",
            "n_lines": len(idx) - 1,
            "input": input_text,   # <-- pass through into manifest
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
    with mp.Pool(processes=procs) as pool, open(out_dir / "manifest.jsonl", "w") as fp:
        for result in tqdm(pool.imap_unordered(index_row, task_iter(), chunksize=chunksize), total=n_docs):
            if result is None or "_error" in result:
                errors += 1
                continue
            fp.write(json.dumps(result) + "\n")

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
        dataset_dir=storage_path / "Nvidia_OpenCodeReasoning",
        out_dir=storage_path / "Nvidia_OpenCodeReasoning_Index",
    )
