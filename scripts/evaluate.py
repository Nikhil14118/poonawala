"""Evaluate the trained detector on the held-out test split and summarise severity output."""
import argparse, json
from collections import Counter
from pathlib import Path
from ultralytics import YOLO
from rdd.severity import severity

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/yolov8n_rdd_india/weights/best.pt")
ap.add_argument("--imgsz", type=int, default=512)
a = ap.parse_args()

m = YOLO(a.weights)
r = m.val(data="configs/rdd_india.yaml", split="test", imgsz=a.imgsz, device="cpu", plots=True, project=str(Path("runs").resolve()), name="eval_test", exist_ok=True)
names = m.names
report = {"mAP50": r.box.map50, "mAP50-95": r.box.map, "precision": r.box.mp, "recall": r.box.mr,
          "per_class": {names[c]: {"P": float(r.box.p[i]), "R": float(r.box.r[i]), "AP50": float(r.box.ap50[i])}
                        for i, c in enumerate(r.box.ap_class_index)}}

sev = Counter()
for res in m.predict("data/yolo/images/test", imgsz=a.imgsz, conf=0.25, device="cpu", stream=True, verbose=False):
    for b in res.boxes:
        _, _, w, h = b.xywhn[0].tolist()
        sev[(names[int(b.cls)], severity(names[int(b.cls)], w, h)[0])] += 1
report["predicted_severity_counts"] = {f"{k[0]}/{k[1]}": v for k, v in sorted(sev.items())}
Path("results").mkdir(exist_ok=True)
Path("results/metrics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
