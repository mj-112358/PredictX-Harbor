"""Perceptual-hash deduplication and grouping.

Every image (and every ``frame_stride``-th video frame) gets a 64-bit pHash of its greyscale content.
Pairs within ``max_distance`` Hamming bits are linked, and so are records sharing a patient ID or a
video. The connected components of that graph are the **groups**: the unit that splits never cut.
A group whose records disagree on lesion class (benign vs malignant) is flagged ``discordant``.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .schema import Record


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def phash(image: np.ndarray) -> int:
    """64-bit DCT perceptual hash of an 8-bit greyscale image."""
    import imagehash
    from PIL import Image

    h = imagehash.phash(Image.fromarray(image), hash_size=8)
    return int(str(h), 16)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between uint64 hash vectors ``a`` (n,) and ``b`` (m,)."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    bytes_ = x.view(np.uint8).reshape(x.shape + (8,))
    return _POP[bytes_].sum(-1)


def near_duplicate_pairs(hashes: np.ndarray, max_distance: int = 6, chunk: int = 2048):
    """All index pairs (i < j) whose hashes differ in at most ``max_distance`` bits."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    n = len(hashes)
    for start in range(0, n, chunk):
        d = hamming_matrix(hashes[start:start + chunk], hashes)
        ii, jj = np.nonzero(d <= max_distance)
        ii = ii + start
        keep = ii < jj
        yield from zip(ii[keep].tolist(), jj[keep].tolist())


def assign_groups(records: Sequence[Record], hashes: Dict[str, int], max_distance: int = 6) -> pd.DataFrame:
    """Group table with columns ``uid, group_id, dup_cluster, discordant, cross_site``.

    ``hashes`` maps uid -> pHash; records without a hash (unsampled video frames) join their video's
    group through the shared patient/video link.
    """
    uids = [r.uid for r in records]
    index = {u: i for i, u in enumerate(uids)}
    uf_dup = UnionFind(len(records))   # duplicates only
    uf_all = UnionFind(len(records))   # duplicates + patient + video
    hashed = [u for u in uids if u in hashes]
    hv = np.array([hashes[u] for u in hashed], dtype=np.uint64)
    for i, j in near_duplicate_pairs(hv, max_distance):
        a, b = index[hashed[i]], index[hashed[j]]
        uf_dup.union(a, b)
        uf_all.union(a, b)
    first: Dict[str, int] = {}
    for i, r in enumerate(records):
        for key in (r.patient_id, r.video_id):
            if key is None:
                continue
            if key in first:
                uf_all.union(first[key], i)
            else:
                first[key] = i
    frame = pd.DataFrame({
        "uid": uids,
        "site": [r.site for r in records],
        "label": [r.label for r in records],
        "group_id": [f"g{uf_all.find(i)}" for i in range(len(records))],
        "dup_cluster": [f"d{uf_dup.find(i)}" for i in range(len(records))],
    })
    lesion = frame[frame.label != "normal"]
    disc = lesion.groupby("dup_cluster").label.nunique()
    frame["discordant"] = frame.dup_cluster.map(disc).fillna(1).gt(1)
    frame["cross_site"] = frame.group_id.map(frame.groupby("group_id").site.nunique()).gt(1)
    return frame


def hash_records(records: Sequence[Record], frame_stride: int = 10, crop_fn=None) -> Dict[str, int]:
    """pHash of every still image and of every ``frame_stride``-th frame of each video."""
    from .loaders.common import read_gray

    out: Dict[str, int] = {}
    for r in records:
        if r.video_id is not None and (r.frame_index or 0) % frame_stride:
            continue
        img = read_gray(r.image_path)
        if crop_fn is not None:
            img = crop_fn(img)
        out[r.uid] = phash(img)
    return out


def summary(groups: pd.DataFrame) -> List[dict]:
    """Per-site counts of near-duplicate clusters, discordant clusters and cross-site links."""
    rows = []
    for site, g in groups.groupby("site"):
        sizes = g.groupby("dup_cluster").size()
        rows.append({"site": site, "records": len(g), "dup_clusters_size_gt1": int((sizes > 1).sum()),
                     "records_in_dup_clusters": int(sizes[sizes > 1].sum()),
                     "discordant_records": int(g.discordant.sum()), "cross_site_records": int(g.cross_site.sum())})
    return rows
