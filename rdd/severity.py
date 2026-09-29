"""Rule-based severity estimation for detected road defects.

RDD2022 has no severity labels, so severity is derived from geometry + defect type:

  extent   - how big the defect is in the image
               pothole / alligator crack : bbox area / image area
               longitudinal / transverse : bbox diagonal / image diagonal   (cracks are thin, so length matters)
  percentile - extent's percentile rank among training-set defects of the SAME class (calibrated once
               from the labels, saved to configs/severity_calibration.json)
  score    - 0.7 * percentile + 0.3 * class_weight
               class weights reflect hazard to vehicles / structural failure:
               pothole 1.0, alligator 0.8, transverse 0.5, longitudinal 0.4
  level    - Low  (score < 0.45), Medium (< 0.70), High (>= 0.70)
"""
import bisect, json
from pathlib import Path

NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
CLASS_WEIGHT = {"longitudinal_crack": 0.4, "transverse_crack": 0.5, "alligator_crack": 0.8, "pothole": 1.0}
AREA_BASED = {"alligator_crack", "pothole"}
LOW_T, HIGH_T = 0.45, 0.70
CAL_PATH = Path(__file__).resolve().parent.parent / "configs" / "severity_calibration.json"


def extent(cls_name, w, h):
    """w, h: normalised (0-1) box width/height."""
    return w * h if cls_name in AREA_BASED else (w * w + h * h) ** 0.5 / 2 ** 0.5


def calibrate(label_dir, out=CAL_PATH):
    """Collect per-class sorted extents from YOLO label files."""
    vals = {n: [] for n in NAMES}
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                n = NAMES[int(p[0])]
                vals[n].append(extent(n, float(p[3]), float(p[4])))
    cal = {n: sorted(v) for n, v in vals.items()}
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(cal))
    return cal


_cal = None


def _load():
    global _cal
    if _cal is None:
        _cal = json.loads(CAL_PATH.read_text())
    return _cal


def severity(cls_name, w, h):
    """Return (level, score) for one box with normalised width/height."""
    vals = _load()[cls_name]
    pct = bisect.bisect_left(vals, extent(cls_name, w, h)) / max(len(vals), 1)
    score = 0.7 * pct + 0.3 * CLASS_WEIGHT[cls_name]
    return ("Low" if score < LOW_T else "Medium" if score < HIGH_T else "High"), round(score, 3)
