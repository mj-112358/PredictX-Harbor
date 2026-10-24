"""BreastVid, the CVA-Net "miccai_buv" video dataset (Lin et al., MICCAI 2022).

Layout of the released ``miccai_buv`` 7-Zip archive: ``rawframes/{benign,malignant}/<video_id>/<frame>.png``
and COCO-VID annotation files (``imagenet_vid_train_15frames.json``, ``imagenet_vid_val.json``; the
README calls them ``train.json`` / ``val.json``) with per-frame lesion boxes ``[x, y, w, h]``. Frames
listed in an annotation file without a box are lesion-free frames of a lesion video; frames present
on disk but listed in no annotation file are kept out (their lesion status is unknown). The
corrections published in the CVA-Net README (2023-03-25) are applied here.

Ingestion: the archive unpacks to 29.5 GB of 692 x 692 PNG frames stored as three identical colour
channels. :func:`ingest` extracts it in batches of videos and re-encodes every frame whose three
channels are identical as a single-channel PNG (lossless; checked pixel by pixel), keeping any frame
with real colour as is. The result is about 5 GB.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from ..schema import Record
from .common import LayoutError, images_under, prepare_root

SITE = "BreastVid"
DROP = {"benign/x66ef02e7f1b9a0ef"}
RELABEL_MALIGNANT = {"benign/x63c9ba1377f35bf6", "benign/x5a1c46ec6377e946"}
SAME_PATIENT = {"benign/x63c9ba1377f35bf6", "benign/x5a1c46ec6377e946", "malignant/2390fbea047347b",
                "malignant/7a39ab5d4970bf89"}


def _video_key(path: str) -> str:
    """'<class>/<video_id>' from any path containing '.../<class>/<video_id>/<frame>'."""
    parts = Path(path.replace("\\", "/")).parts
    for i, part in enumerate(parts[:-2]):
        if part.lower() in ("benign", "malignant"):
            return f"{part.lower()}/{parts[i + 1]}"
    raise LayoutError(f"BreastVid: cannot find class/video in {path!r}")


def _frame_index(name: str) -> int:
    digits = re.findall(r"\d+", Path(name).stem)
    return int(digits[-1]) if digits else 0


def _safe_member(name: str) -> bool:
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts


def ingest(archive: Path, out: Path, batch: int = 20) -> dict:
    """Extract the 7-Zip archive in batches of videos with lossless single-channel re-encoding."""
    marker = out / ".ingested.json"
    if marker.exists():
        return json.loads(marker.read_text())
    listing = subprocess.run(["7z", "l", "-slt", str(archive)], capture_output=True, text=True, check=True).stdout
    members = [line[7:] for line in listing.splitlines() if line.startswith("Path = ")][1:]
    if not all(_safe_member(m) for m in members):
        raise LayoutError("BreastVid: unsafe member path in archive")
    videos = sorted({"/".join(m.split("/")[:3]) for m in members if m.startswith("rawframes/") and m.count("/") >= 3})
    tmp = out.parent / (out.name + "_tmp")
    out.mkdir(parents=True, exist_ok=True)
    stats = {"frames": 0, "converted_to_grey": 0, "kept_colour": 0, "videos": len(videos)}
    subprocess.run(["7z", "x", "-y", f"-o{out}", str(archive), "*.json"], check=True, capture_output=True)
    for s in range(0, len(videos), batch):
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["7z", "x", "-y", f"-o{tmp}", str(archive), *[v + "/*" for v in videos[s:s + batch]]],
                       check=True, capture_output=True)
        for f in sorted(tmp.rglob("*.png")):
            rel = f.relative_to(tmp)
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img.ndim == 3 and np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]):
                cv2.imwrite(str(dest), img[..., 0], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                stats["converted_to_grey"] += 1
            else:
                shutil.copyfile(f, dest)
                stats["kept_colour"] += 1
            stats["frames"] += 1
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[ingest] BreastVid {min(s + batch, len(videos))}/{len(videos)} videos", flush=True)
    marker.write_text(json.dumps(stats))
    return stats


def load(raw_dir: str, extracted_dir: str) -> List[Record]:
    archives = sorted(Path(raw_dir).glob("*.7z"))
    if archives:
        ingest(archives[0], Path(extracted_dir))
        root = Path(extracted_dir)
    else:
        root = prepare_root(raw_dir, extracted_dir)
    jsons = [p for p in root.rglob("*.json") if not p.name.startswith(".") and "images" in json.loads(p.read_text())]
    if not jsons:
        raise LayoutError(f"BreastVid: no COCO annotation file under {root}")
    frames = {str(p.relative_to(root)).replace("\\", "/"): p for p in images_under(root)}
    by_suffix: Dict[str, Path] = {}
    for rel, p in frames.items():
        key = _video_key(rel) + "/" + p.name
        by_suffix[key] = p

    boxes: Dict[str, list] = {}
    split_of: Dict[str, str] = {}
    for js in jsons:
        coco = json.loads(js.read_text())
        id_to_key = {}
        for im in coco["images"]:
            key = _video_key(im["file_name"]) + "/" + Path(im["file_name"]).name
            id_to_key[im["id"]] = key
            boxes.setdefault(key, [])
            split_of[key] = "val" if "val" in js.stem else "train"
        for an in coco.get("annotations", []):
            x, y, w, h = an["bbox"]
            if w > 0 and h > 0:
                boxes[id_to_key[an["image_id"]]].append((round(x), round(y), round(x + w), round(y + h)))

    records: List[Record] = []
    for key, bxs in sorted(boxes.items()):
        video = key.rsplit("/", 1)[0]
        if video in DROP:
            continue
        path = by_suffix.get(key)
        if path is None:
            raise LayoutError(f"BreastVid: annotated frame {key} not found on disk")

        label = "malignant" if video in RELABEL_MALIGNANT else video.split("/")[0]
        patient = "corrected_group_1" if video in SAME_PATIENT else video.replace("/", "_")
        records.append(Record(uid=f"{SITE}:{key}", site=SITE, image_path=str(path), label=label,
                              patient_id=f"{SITE}:{patient}", boxes=bxs, lesion_id=f"{SITE}:{video}",
                              video_id=f"{SITE}:{video}", frame_index=_frame_index(path.name),
                              meta={"official_split": split_of[key]}))
    return records
