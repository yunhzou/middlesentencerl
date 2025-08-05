#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union
import argparse, json, logging, random, torch, numpy as np, mmap, os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# --------- fast I/O helpers ----------
def _read_text_mmap(path: Path) -> str:
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.read().decode("utf-8", errors="strict")

def _memmap_idx(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint32, mode="r")

# --------- dataset ----------
class ContiguousMiddleSpanDataset(torch.utils.data.IterableDataset):
    """
    Stream samples where the *target* is a contiguous block of exactly `mid_len` sentences
    taken from the document, and the *context* is the full document text.

    Parameters
    ----------
    mid_len : int
        Exact number of sentences in the target span.
    samples_per_doc : int | "full"
        * int  – draw ≤ that many random spans per doc (without replacement).
        * "full" – enumerate every valid span of length `mid_len`.
    index_dir : str | Path
        Folder with manifest.jsonl and .txt/.idx files.
    seed : int
        Base RNG seed (each DataLoader worker gets seed + worker_id).
    shuffle_docs : bool
        Shuffle document order per-epoch/iterator.
    return_mode : "full" | "triples"
        "full" returns the whole context; "triples" returns prev/target/post only.
    """
    def __init__(
        self,
        mid_len: int = 3,
        samples_per_doc: Union[int, str] = 3,
        index_dir: Union[str, Path] = "olmo_char_demo",
        seed: int = 0,
        shuffle_docs: bool = True,
        return_mode: str = "full",
    ):
        if not (isinstance(mid_len, int) and mid_len > 0):
            raise ValueError("mid_len must be a positive int")
        if not isinstance(samples_per_doc, (int, str)) or (isinstance(samples_per_doc, int) and samples_per_doc <= 0):
            raise ValueError("samples_per_doc must be >0 int or 'full'")
        if return_mode not in ("full", "triples"):
            raise ValueError("return_mode must be 'full' or 'triples'")

        self.mid_len = mid_len
        self.samples_per_doc = samples_per_doc
        self.index_dir = Path(index_dir)
        self.seed = seed
        self.shuffle_docs = shuffle_docs
        self.return_mode = return_mode

        mpath = self.index_dir / "manifest.jsonl"
        if not mpath.is_file():
            raise FileNotFoundError(mpath)

        self.meta: List[Dict] = []
        with mpath.open("r") as fp:
            for line in fp:
                m = json.loads(line)
                self.meta.append({"txt": m["txt"], "idx": m["idx"], "n_sent": int(m["n_sent"])})
        if not self.meta:
            raise RuntimeError("Manifest is empty")

        # per-process caches
        self._idx_cache: Dict[str, np.memmap] = {}
        self._txt_cache: Dict[str, str] = {}
        self._cached_len: Union[int, None] = None

    # ----- helpers -----
    def _get_idx(self, fname: str) -> np.memmap:
        p = str(self.index_dir / fname)
        arr = self._idx_cache.get(p)
        if arr is None:
            arr = _memmap_idx(Path(p))
            self._idx_cache[p] = arr
        return arr

    def _get_txt(self, fname: str) -> str:
        p = str(self.index_dir / fname)
        txt = self._txt_cache.get(p)
        if txt is None:
            txt = _read_text_mmap(Path(p))
            self._txt_cache[p] = txt
        return txt

    @staticmethod
    def _span_char_bounds(idx: np.ndarray, s: int, L: int) -> Tuple[int, int]:
        return int(idx[s]), int(idx[s + L])

    def _valid_starts(self, n_sent: int) -> int:
        # # valid starts for fixed L = max(0, n - L + 1)
        return max(0, n_sent - self.mid_len + 1)

    def _samples_from_doc(self, meta: Dict, rng: random.Random):
        doc_txt = meta["txt"]
        try:
            idx = self._get_idx(meta["idx"])
            text = self._get_txt(doc_txt)
        except Exception as e:
            logging.warning(f"Skip {doc_txt}: I/O error → {e}")
            return

        n_sent = meta["n_sent"]
        n_valid = self._valid_starts(n_sent)
        if n_valid == 0:
            return

        if self.samples_per_doc == "full":
            starts = range(n_valid)  # 0..n_valid-1
        else:
            k = min(int(self.samples_per_doc), n_valid)
            starts = rng.sample(range(n_valid), k)  # without replacement

        L = self.mid_len
        if self.return_mode == "triples":
            for s in starts:
                c0, c1 = self._span_char_bounds(idx, int(s), L)
                yield {
                    "prev":  text[:c0].rstrip(),
                    "target": text[c0:c1].strip(),
                    "post":  text[c1:].lstrip(),
                    "target_char_start": int(c0),
                    "target_char_end":   int(c1),
                    "target_sent_start": int(s),
                    "target_sent_end":   int(s)+L,
                }
        else:
            for s in starts:
                c0, c1 = self._span_char_bounds(idx, int(s), L)
                yield {
                    "context": text,
                    "target":  text[c0:c1].strip(),
                    "target_char_start": int(c0),
                    "target_char_end":   int(c1),
                    "target_sent_start": int(s),
                    "target_sent_end":   int(s)+L,
                }

    # ----- IterableDataset API -----
    def __iter__(self) -> Iterable[Dict[str, Union[str, int]]]:
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        rng = random.Random(self.seed + worker_id)

        N = len(self.meta)
        order = list(range(N))
        if self.shuffle_docs:
            rng.shuffle(order)
        shard = order[worker_id::num_workers]

        for i in shard:
            for samp in self._samples_from_doc(self.meta[i], rng):
                yield samp

    def __len__(self) -> int:
        if self._cached_len is not None:
            return self._cached_len
        total = 0
        L = self.mid_len
        for m in self.meta:
            n = m["n_sent"]
            n_valid = max(0, n - L + 1)
            if self.samples_per_doc == "full":
                total += n_valid
            else:
                total += min(int(self.samples_per_doc), n_valid)
        self._cached_len = int(total)
        return self._cached_len

# ---------------- demo ----------------
def demo():
    p = argparse.ArgumentParser(description="Single-length ContiguousMiddleSpanDataset demo")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--mid_len", type=int, default=2)
    p.add_argument("--per_doc", type=str, default="2")  # "full" or int
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--return_mode", type=str, choices=["full","triples"], default="full")
    args = p.parse_args()

    per_doc: Union[int, str] = "full" if args.per_doc == "full" else int(args.per_doc)
    storage_path = Path(os.getenv("STORAGE_PATH"))
    ds = ContiguousMiddleSpanDataset(
        mid_len=args.mid_len,
        samples_per_doc=per_doc,
        index_dir=storage_path / "AoPS-Instruct_Index",
        seed=args.seed,
        return_mode=args.return_mode,
    )

    print(f"Dataset size (computed): {len(ds):,} samples")

    it = iter(ds)
    for i in range(1, args.n_samples + 1):
        ex = next(it)
        if args.return_mode == "triples":
            print(f"\n===== SAMPLE {i} =====")
            print(f"chars: [{ex['target_char_start']}..{ex['target_char_end']}) "
                  f"sent_span: [{ex['target_sent_start']}..{ex['target_sent_end']})")
            print("PREV   :", ex["prev"])
            print("TARGET :", ex["target"])
            print("POST   :", ex["post"])
        else:
            ctx = ex["context"]; s0, s1 = ex["target_char_start"], ex["target_char_end"]
            prev, targ, post = ctx[:s0].rstrip(), ctx[s0:s1].strip(), ctx[s1:].lstrip()
            print(f"\n===== SAMPLE {i} =====")
            print(f"chars: [{s0}..{s1}) sent_span: [{ex['target_sent_start']}..{ex['target_sent_end']})")
            print("PREV   :", prev)
            print("TARGET :", targ)
            print("POST   :", post)

if __name__ == "__main__":
    demo()
