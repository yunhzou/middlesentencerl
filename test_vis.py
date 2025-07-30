#!/usr/bin/env python
"""
char_middle_loader_fixed.py
---------------------------
MiddleSentenceDataset (fixed‑window version)
• `n_prev`  and `n_post` are **integers** (exact window sizes).
• Returns triples {"prev", "target", "post"} as strings.
"""

import json, random, argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import IterableDataset


class MiddleSentenceDataset(IterableDataset):
    def __init__(self,
                 manifest: Path,
                 n_prev:   int = 1,
                 n_middle: int = 1,
                 n_post:   int = 0,
                 full:     bool = False,
                 seed:     int = 0,
                 index_dir: str = "olmo_char_demo"):
        self.n_prev  = n_prev
        self.n_mid   = n_middle
        self.n_post  = n_post
        self.full    = full
        self.seed    = seed
        self.index_dir = index_dir
        self.meta    = [json.loads(l) for l in open(manifest, "r")]

    # ---------- helpers ------------------------------------------------ #
    @staticmethod
    def _load_idx(path: Path) -> np.ndarray:
        return np.fromfile(path, dtype=np.uint32)

    @staticmethod
    def _load_txt(path: Path) -> str:
        return Path(path).read_text(encoding="utf-8")

    # ---------- iterate over one document ------------------------------ #
    def _iter_doc(self, meta: Dict, rng: random.Random):
        idx  = self._load_idx(Path(self.index_dir)/Path(meta["idx"]))
        text = self._load_txt(Path(self.index_dir)/Path(meta["txt"]))

        n_sent = len(idx) - 1
        start_min = self.n_prev
        start_max = n_sent - self.n_mid - self.n_post
        if start_min > start_max:
            return  # document too short

        starts = range(start_min, start_max + 1) if self.full \
                 else (rng.randint(start_min, start_max),)

        for s in starts:  # s = idx of the first middle sentence
            prev   = text[idx[s - self.n_prev]          : idx[s]]
            target = text[idx[s]                        : idx[s + self.n_mid]]
            post   = text[idx[s + self.n_mid]           : idx[s + self.n_mid + self.n_post]]
            yield {"prev": prev.strip(),
                   "target": target.strip(),
                   "post": post.strip()}

    # ---------- dataset iterator --------------------------------------- #
    def __iter__(self) -> Iterable[Dict[str, str]]:
        worker = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker else 0))
        while True:
            yield from self._iter_doc(rng.choice(self.meta), rng)


# -------------------- CLI demo ----------------------------------------- #
def main(manifest: Path, n_samples: int,
         n_prev, n_mid, n_post, full):
    ds = MiddleSentenceDataset(
            manifest,
            n_prev   = n_prev,
            n_middle = n_mid,
            n_post   = n_post,
            full     = full,
            seed     = 123)

    for i, ex in zip(range(1, n_samples + 1), ds):
        print(f"\n===== SAMPLE {i} =====")
        print("PREV   :", ex["prev"])
        print("TARGET :", ex["target"])
        print("POST   :", ex["post"])
        if i == n_samples:
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=Path("olmo_char_demo/manifest.jsonl"))
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--prev",  type=int, default=2)
    p.add_argument("--mid",   type=int, default=1)
    p.add_argument("--post",  type=int, default=2)
    p.add_argument("--full",  action="store_true",
                   help="yield all valid triples per doc instead of sampling one")
    cfg = p.parse_args()

    main(cfg.manifest, cfg.n_samples, cfg.prev, cfg.mid, cfg.post, cfg.full)
