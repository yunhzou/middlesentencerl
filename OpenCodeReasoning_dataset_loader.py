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
    Target is a contiguous block of exactly `mid_len` LINES taken from the SOLUTION
    (indexed by a line-start .idx). Context is INPUT + SOLUTION merged at load time.

    samples_per_doc: int or "full" (enumerate all valid mid_len windows in solution)
    return_mode: "full" -> returns full merged context; "triples" -> prev/target/post
    """
    def __init__(
        self,
        mid_len: int = 3,
        samples_per_doc: Union[int, str] = 3,
        index_dir: Union[str, Path] = "AoPS-Instruct_Index",
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
                # Expect n_lines; fallback to n_sent if old manifest
                n_lines = int(m.get("n_lines", m.get("n_sent")))
                self.meta.append({
                    "id":  m.get("id", Path(m["txt"]).stem),
                    "txt": m["txt"],       # solution text file
                    "idx": m["idx"],       # line-start memmap for solution
                    "n_lines": n_lines,    # number of line boundaries + 1
                    "input": m.get("input", ""),   # <--- include input here
                })
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

    def _get_input_sidecar(self, doc_id: str) -> str:
        """
        Try to read an optional INPUT sidecar named '{doc_id}.input.txt'.
        If not present, return ''.
        """
        p = self.index_dir / f"{doc_id}.input.txt"
        if p.is_file():
            return self._get_txt(str(p.relative_to(self.index_dir)))
        return ""

    @staticmethod
    def _span_char_bounds(idx: np.ndarray, s: int, L: int) -> Tuple[int, int]:
        # On SOLUTION only (indices are line-starts in solution)
        return int(idx[s]), int(idx[s + L])

    def _valid_starts(self, n_lines: int) -> int:
        # Valid line-window starts for fixed L
        return max(0, n_lines - self.mid_len)

    def _samples_from_doc(self, meta: Dict, rng: random.Random):
        try:
            idx = self._get_idx(meta["idx"])
            solution = self._get_txt(meta["txt"])
        except Exception as e:
            logging.warning(f"Skip {meta['txt']}: I/O error → {e}")
            return

        # build merged context
        input_text = meta.get("input", "")
        prefix = f"Question:\n{input_text}\n"
        context = prefix + solution
        offset = len(prefix)

        n_lines = meta["n_lines"]
        n_valid = max(0, n_lines - self.mid_len)
        if n_valid == 0:
            return

        if self.samples_per_doc == "full":
            starts = range(n_valid)
        else:
            k = min(int(self.samples_per_doc), n_valid)
            starts = rng.sample(range(n_valid), k)

        L = self.mid_len

        for s in starts:
            c0_sol, c1_sol = self._span_char_bounds(idx, int(s), L)
            c0, c1 = c0_sol + offset, c1_sol + offset
            if self.return_mode == "triples":
                yield {
                    "input":  input_text,
                    "prev":   context[offset:c0].rstrip(),
                    "target": context[c0:c1].strip(),
                    "post":   context[c1:].lstrip(),
                    "context": context,
                    "target_char_start": int(c0),
                    "target_char_end":   int(c1),
                    "target_line_start": int(s),
                    "target_line_end":   int(s) + L,
                    "doc_id": meta["id"],
                }
            else:
                yield {
                    "input":  input_text,
                    "context": context,
                    "target":  context[c0:c1].strip(),
                    "target_char_start": int(c0),
                    "target_char_end":   int(c1),
                    "target_line_start": int(s),
                    "target_line_end":   int(s) + L,
                    "doc_id": meta["id"],
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
            yield from self._samples_from_doc(self.meta[i], rng)

    def __len__(self) -> int:
        if self._cached_len is not None:
            return self._cached_len
        total = 0
        L = self.mid_len
        for m in self.meta:
            n = m["n_lines"]
            n_valid = max(0, n - L)
            if self.samples_per_doc == "full":
                total += n_valid
            else:
                total += min(int(self.samples_per_doc), n_valid)
        self._cached_len = int(total)
        return self._cached_len

# ---------------- demo ----------------
def demo():
    n_samples_for_demo = 10
    return_mode = "triples"
    ds = ContiguousMiddleSpanDataset(
        mid_len=2,
        samples_per_doc=2, # or "full"
        index_dir="/home/yunhengzou/middlesentencerldata/Nvidia_OpenCodeReasoning_Index",
        seed=123,
        return_mode=return_mode,
    )

    print(f"Dataset size (computed): {len(ds):,} samples")

    it = iter(ds)
    for i in range(1, n_samples_for_demo + 1):
        ex = next(it)
        if return_mode == "triples":
            print(f"\n===== SAMPLE {i} ({ex['doc_id']}) =====")
            print(f"INPUT  : {ex['input']}")   # <--- NEW
            print(f"chars: [{ex['target_char_start']}..{ex['target_char_end']}) "
                f"line_span: [{ex['target_line_start']}..{ex['target_line_end']})")
            print("PREV   :", ex["prev"])
            print("TARGET :", ex["target"])
            print("POST   :", ex["post"])
        else:
            ctx = ex["context"]; s0, s1 = ex["target_char_start"], ex["target_char_end"]
            prev, targ, post = ctx[:s0].rstrip(), ctx[s0:s1].strip(), ctx[s1:].lstrip()
            print(f"\n===== SAMPLE {i} ({ex['doc_id']}) =====")
            print(f"INPUT  : {ex['input']}")   # <--- NEW
            print(f"chars: [{s0}..{s1}) line_span: [{ex['target_line_start']}..{ex['target_line_end']})")
            print("PREV   :", prev)
            print("TARGET :", targ)
            print("POST   :", post)

if __name__ == "__main__":
    demo()