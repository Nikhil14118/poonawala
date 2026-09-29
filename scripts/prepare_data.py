"""Convert RDD2022 India (Pascal VOC XML) to a YOLO dataset with a seeded train/val/test split.

Official RDD2022 test labels are hidden, so we split the labelled `train` folder ourselves.
"""
import random, shutil, zipfile, sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "yolo"
CLASSES = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def main():
    src = DATA / "RDD2022" / "India"
    if not src.exists():
        with zipfile.ZipFile(DATA / "India.zip") as z:
            z.extractall(DATA / "RDD2022")
        if not src.exists():  # some zips already contain the top-level India/ folder
            src = DATA / "RDD2022"
    xmls = sorted(src.rglob("annotations/xmls/*.xml")) or sorted(src.rglob("*.xml"))
    print("annotation files:", len(xmls))

    samples, skipped = [], Counter()
    for x in xmls:
        r = ET.parse(x).getroot()
        w = int(r.findtext("size/width")); h = int(r.findtext("size/height"))
        lines = []
        for o in r.iter("object"):
            name = o.findtext("name").strip()
            if name not in CLASSES:
                skipped[name] += 1; continue
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, x2 = max(0, x1), min(w, x2); y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{CLASSES[name]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        img = next((p for p in (x.parent.parent.parent / "images" / (x.stem + e) for e in (".jpg", ".JPG", ".png"))
                    if p.exists()), None)
        if img is not None:
            samples.append((img, lines))
    print("usable images:", len(samples), "skipped classes:", dict(skipped))

    random.Random(SEED).shuffle(samples)
    n = len(samples)
    cut = {"train": int(n * .8), "val": int(n * .9), "test": n}
    bounds = {"train": (0, cut["train"]), "val": (cut["train"], cut["val"]), "test": (cut["val"], n)}
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = {}
    for sp, (a, b) in bounds.items():
        (OUT / "images" / sp).mkdir(parents=True); (OUT / "labels" / sp).mkdir(parents=True)
        c = Counter()
        for img, lines in samples[a:b]:
            shutil.copy2(img, OUT / "images" / sp / img.name)
            (OUT / "labels" / sp / (img.stem + ".txt")).write_text("\n".join(lines))
            c.update(int(l.split()[0]) for l in lines)
        stats[sp] = (b - a, {NAMES[k]: v for k, v in sorted(c.items())})
        print(sp, stats[sp])

    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs" / "rdd_india.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))


if __name__ == "__main__":
    main()
