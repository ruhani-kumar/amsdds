import argparse, glob, os, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocessing import shades_of_gray, make_splits


def main(out_dir, cache_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    import kagglehub
    data = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    print("dataset:", data)

    meta = glob.glob(f"{data}/**/HAM10000_metadata*.csv", recursive=True)[0]
    df = pd.read_csv(meta)

    src = {os.path.splitext(os.path.basename(p))[0]: p
           for p in glob.glob(f"{data}/**/*.jpg", recursive=True)}
    df["src"] = df.image_id.map(src)
    assert df.src.isna().sum() == 0, "some images not found"

    df = make_splits(df)
    df["path"] = df.image_id.map(lambda i: os.path.join(cache_dir, f"{i}.jpg"))
    df.to_csv(os.path.join(out_dir, "splits.csv"), index=False)
    print(df.split.value_counts().to_dict())
    print(pd.crosstab(df.dx, df.split))

    def build(row):
        if os.path.exists(row.path):
            return
        img = Image.open(row.src).convert("RGB")
        w, h = img.size
        s = 256 / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        img = Image.fromarray(shades_of_gray(np.array(img)))
        img.save(row.path, quality=95)

    todo = [r for r in df.itertuples() if not os.path.exists(r.path)]
    if todo:
        t0 = time.time()
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(build, todo))
        print(f"cached {len(todo)} images in {time.time()-t0:.0f}s")
    else:
        print("image cache already built")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data")
    ap.add_argument("--cache", default="/tmp/ham_cache")
    a = ap.parse_args()
    main(a.out, a.cache)
