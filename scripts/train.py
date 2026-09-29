"""Fine-tune a lightweight YOLOv8n detector on RDD2022-India (CPU friendly defaults)."""
import argparse
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--imgsz", type=int, default=512)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
ap.add_argument("--name", default="yolov8n_rdd_india")
ap.add_argument("--weights", default="yolov8n.pt", help="starting weights")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

if a.resume:
    YOLO(f"runs/{a.name}/weights/last.pt").train(resume=True)
else:
    YOLO(a.weights).train(
        data="configs/rdd_india.yaml", epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        fraction=a.fraction, device="cpu", workers=6, project=str(Path("runs").resolve()), name=a.name,
        exist_ok=True, patience=10, cos_lr=True, seed=0, plots=True,
    )
