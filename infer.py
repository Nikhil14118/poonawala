"""Run road-damage detection + severity on an image or folder.

    python infer.py path/to/image_or_folder [--weights best.pt] [--out results/predictions]
Outputs annotated images and a predictions.json with class, confidence, bbox (xyxy px) and severity.
"""
import argparse, json
from pathlib import Path
import cv2
from ultralytics import YOLO
from rdd.severity import severity

COLORS = {"Low": (60, 200, 60), "Medium": (0, 165, 255), "High": (40, 40, 230)}  # BGR

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("--weights", default="weights/best.pt")
ap.add_argument("--out", default="results/predictions")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=416)
a = ap.parse_args()

model = YOLO(a.weights)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
records = {}
for r in model.predict(a.source, conf=a.conf, imgsz=a.imgsz, device="cpu", stream=True, verbose=False):
    img, dets = r.orig_img.copy(), []
    for b in r.boxes:
        name = model.names[int(b.cls)]
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        _, _, w, h = b.xywhn[0].tolist()
        level, score = severity(name, w, h)
        dets.append({"class": name, "confidence": round(float(b.conf), 3), "bbox_xyxy": [x1, y1, x2, y2],
                     "severity": level, "severity_score": score})
        c = COLORS[level]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"{name} {b.conf.item():.2f} | {level}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    p = Path(r.path); cv2.imwrite(str(out / p.name), img); records[p.name] = dets
(out / "predictions.json").write_text(json.dumps(records, indent=2))
print(f"{len(records)} image(s) processed -> {out}")
