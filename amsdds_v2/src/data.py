"""HAM10000 manifest + dataset. Colour constancy is applied exactly as v1 does:
shades-of-gray p=6 on the ORIGINAL PIL image, once, before the model transform.
Skipping it is silent train/serve skew against the deployed Layer 1.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MALIGNANT = ["mel", "bcc", "akiec"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


def shades_of_gray(arr: np.ndarray, power: int = 6) -> np.ndarray:
    a = arr.astype(np.float32)
    v = np.power(np.mean(np.power(a, power), (0, 1)), 1.0 / power)
    v = v / (np.sqrt((v ** 2).sum()) + 1e-8)
    return np.clip(a / (v * np.sqrt(3) + 1e-8), 0, 255).astype(np.uint8)


def index_images(*roots: str) -> dict[str, str]:
    """image_id -> absolute path, searched recursively under each root."""
    out = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for ext in ("jpg", "jpeg", "png"):
            for p in glob.glob(f"{root}/**/*.{ext}", recursive=True):
                out.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return out


def _pick(df: pd.DataFrame, *names: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def build_manifest(splits_csv: str, image_roots: list[str]) -> pd.DataFrame:
    """Join your existing lesion-grouped split against the images on disk.

    REUSE the v1 splits.csv. Re-splitting with a different seed makes every
    number here incomparable to the 0.8446 / 0.857 baseline you're trying to
    beat, and you lose the ability to claim an improvement at all.
    """
    df = pd.read_csv(splits_csv)
    c_img = _pick(df, "image_id", "image", "img_id")
    c_dx = _pick(df, "dx", "label", "diagnosis")
    c_split = _pick(df, "split", "fold", "set")

    paths = index_images(*image_roots)
    df = df.rename(columns={c_img: "image_id", c_dx: "dx", c_split: "split"})
    df["path"] = df["image_id"].map(paths)

    missing = int(df["path"].isna().sum())
    if missing:
        print(f"[data] WARNING {missing} rows have no image on disk — dropping")
        df = df.dropna(subset=["path"])

    df["y"] = df["dx"].map(CLS2IDX)
    if df["y"].isna().any():
        bad = sorted(df.loc[df["y"].isna(), "dx"].unique())
        raise ValueError(f"unmapped labels {bad}; expected {CLASSES}")
    df["y"] = df["y"].astype(int)

    print(f"[data] {len(df)} images")
    print(df.groupby(["split", "dx"]).size().unstack(fill_value=0))
    return df.reset_index(drop=True)


class LesionDS(Dataset):
    """Returns (tensor, label). `views` > 1 stacks flip/rotate TTA views, in
    which case the item tensor is [views, 3, H, W]."""

    def __init__(self, df: pd.DataFrame, transform, colour_constancy: bool = True,
                 power: int = 6, views: int = 1):
        self.paths = df["path"].tolist()
        self.labels = df["y"].tolist()
        self.tf = transform
        self.cc = colour_constancy
        self.power = power
        self.views = max(1, min(4, views))

    def __len__(self):
        return len(self.paths)

    def _views(self, img: Image.Image):
        v = [img,
             img.transpose(Image.FLIP_LEFT_RIGHT),
             img.transpose(Image.FLIP_TOP_BOTTOM),
             img.transpose(Image.ROTATE_90)]
        return v[:self.views]

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.cc:
            img = Image.fromarray(shades_of_gray(np.array(img), self.power))
        xs = [self.tf(v) for v in self._views(img)]
        x = xs[0] if self.views == 1 else torch.stack(xs)
        return x, self.labels[i]


def class_prior(df: pd.DataFrame, split: str = "train") -> np.ndarray:
    """Empirical prior over CLASSES, in CLASSES order."""
    counts = df[df["split"] == split]["y"].value_counts().reindex(range(len(CLASSES)), fill_value=0)
    p = counts.to_numpy(dtype=np.float64)
    return (p / p.sum()).astype(np.float32)
