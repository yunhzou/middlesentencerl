from __future__ import annotations
from pathlib import Path
from typing   import Dict, Iterable, List, Union
import argparse, json, random, torch, numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,               # default → INFO
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S")



class MiddleSentenceDataset(torch.utils.data.IterableDataset):
    """
    Stream (prev, target, post) triples, centred on `n_mid` sentences.

    Parameters
    ----------
    manifest : Path
        JSONL file – one line per document with keys
        {"txt": "rel/path/to/text.txt", "idx": "rel/path/to/text.idx"}.
    n_prev, n_mid, n_post : int
        # sentences before, inside, and after the target window.
    samples_per_doc : int | "full"
        * int  – draw ≤ that many random windows per document.
        * "full" – enumerate every valid window.
    index_dir : str | Path
        Root folder containing all txt/idx files.
    seed : int
        Base RNG seed (each DataLoader worker gets seed + worker_id).
    """

    def __init__(
        self,
        manifest: Path,
        n_prev: int = 1,
        n_mid: int = 1,
        n_post: int = 0,
        samples_per_doc: Union[int, str] = 1,
        index_dir: Union[str, Path] = "olmo_char_demo",
        seed: int = 0,
    ):
        assert n_prev >= 0 and n_mid > 0 and n_post >= 0
        if isinstance(samples_per_doc, int) and samples_per_doc <= 0:
            raise ValueError("samples_per_doc must be > 0")
        if not isinstance(samples_per_doc, (int, str)):
            raise TypeError("samples_per_doc must be int or 'full'")

        self.n_prev = n_prev
        self.n_mid = n_mid
        self.n_post = n_post
        self.samples_per_doc = samples_per_doc
        self.index_dir = Path(index_dir)
        self.seed = seed

        mpath = Path(manifest)
        if not mpath.is_file():
            raise FileNotFoundError(mpath)
        self.meta: List[Dict] = [json.loads(l) for l in mpath.read_text().splitlines()]
        if not self.meta:
            raise RuntimeError("Manifest is empty")

    # ────────────── helpers ────────────────────────────────────────────
    @staticmethod
    def _load_idx(path: Path) -> np.ndarray:
        return np.fromfile(path, dtype=np.uint32)

    @staticmethod
    def _load_txt(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    # ────────────── per-document sampling ─────────────────────────────
    def _samples_from_doc(
        self, meta: Dict, rng: random.Random
    ) -> Iterable[Dict[str, str]]:
        doc_txt = meta.get("txt", "<?>")
        try:
            idx = self._load_idx(self.index_dir / meta["idx"])
            text = self._load_txt(self.index_dir / meta["txt"])
        except Exception as e:
            logging.warning(f"Skip {doc_txt}: I/O error → {e}")
            return

        n_sent = len(idx) - 1
        need = self.n_prev + self.n_mid + self.n_post
        start_min, start_max = self.n_prev, n_sent - self.n_mid - self.n_post

        if n_sent < need or start_min > start_max:
            logging.info(f"Skip {doc_txt}: only {n_sent} sentences, need ≥ {need}")
            return

        valid = list(range(start_min, start_max + 1))
        if self.samples_per_doc == "full":
            chosen = valid
        else:  # int
            k = min(self.samples_per_doc, len(valid))
            chosen = rng.sample(valid, k)

        for s in chosen:
            prev = text[idx[s - self.n_prev] : idx[s]]
            tgt = text[idx[s] : idx[s + self.n_mid]]
            post = text[idx[s + self.n_mid] : idx[s + self.n_mid + self.n_post]]
            yield {"prev": prev.strip(), "target": tgt.strip(), "post": post.strip()}

    # ────────────── iterator required by IterableDataset ──────────────
    def __iter__(self) -> Iterable[Dict[str, str]]:
        """Yield samples from *each* document once, then stop."""
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))

        # Make a private, shuffled copy of the manifest
        doc_queue = self.meta.copy()
        rng.shuffle(doc_queue)

        for meta in doc_queue:                      # walk each doc exactly once
            yielded = False
            for samp in self._samples_from_doc(meta, rng):
                yielded = True
                yield samp
            if not yielded:
                continue  

    def __len__(self) -> int:
        """Exact size, accounting for skips and samples_per_doc."""
        if hasattr(self, "_cached_len"):
            return self._cached_len

        n_total = 0
        need = self.n_prev + self.n_mid + self.n_post

        for meta in self.meta:
            try:
                idx = self._load_idx(self.index_dir / meta["idx"])
            except Exception:
                continue  # unreadable → contributes nothing
            n_sent = len(idx) - 1
            if n_sent < need:
                continue  # too short

            n_valid = n_sent - need + 1           # all legal windows
            if self.samples_per_doc == "full":
                n_total += n_valid
            else:
                n_total += min(self.samples_per_doc, n_valid)

        self._cached_len = n_total                # memoise
        return n_total

# ---------------------------------------------------------------------- #
def demo():
    p = argparse.ArgumentParser(description="Visual sanity-check for MiddleSentenceDataset")
    p.add_argument("--manifest", type=Path, default="olmo_char_demo/manifest.jsonl")
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--prev",   type=int, default=1)
    p.add_argument("--mid",    type=int, default=1)
    p.add_argument("--post",   type=int, default=1)
    p.add_argument("--per_doc", type=str,
                   default="1",
                   help="'full' or an integer (e.g. 3) → how many windows per document")
    args = p.parse_args()

    per_doc = "full" if args.per_doc.lower() == "full" else int(args.per_doc)

    ds = MiddleSentenceDataset(
            manifest         = args.manifest,
            n_prev           = args.prev,
            n_mid            = args.mid,
            n_post           = args.post,
            samples_per_doc  = 5,
            seed             = 123)
    # print the size of the dataset
    print(f"Dataset size: {len(ds):,} samples")
    from itertools import count
    real_n = sum(1 for _ in ds)
    print(real_n)

    for i, ex in zip(range(1, args.n_samples + 1), ds):
        print(f"\n===== SAMPLE {i} =====")
        print("PREV   :", ex["prev"])
        print("TARGET :", ex["target"])
        print("POST   :", ex["post"])

if __name__ == "__main__":
    demo()