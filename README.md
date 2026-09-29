# Road Damage Detection & Severity Assessment (Case 3)

Lightweight CV system on the **India subset of RDD2022** (7,706 smartphone road images) that
detects defects, localises them with boxes, classifies them and estimates severity.

| Output | How |
|---|---|
| Box + class (pothole, longitudinal, transverse, alligator crack) | YOLOv8n fine-tuned on RDD2022-India (labels D40, D00, D10, D20) |
| Severity (Low / Medium / High) | Rule-based score from defect size + defect type (see below) |

## Pipeline
1. `scripts/download_india.py` – pulls only `India.zip` (~500 MB) out of the 13 GB Figshare archive.
2. `scripts/prepare_data.py` – VOC XML -> YOLO labels, seeded 80/10/10 split (6,164 / 771 / 771 images; official test labels are hidden).
   Other RDD labels (D44, D01, D11, D43, D50) are dropped, so images with only those act as background.
3. `scripts/train.py` – YOLOv8n fine-tuning (CPU only).
4. `scripts/evaluate.py` – mAP / P / R on the held-out test split + predicted severity counts (`results/metrics.json`).
5. `infer.py` – `python infer.py <image_or_folder> --weights weights/best.pt --imgsz 416` -> annotated images + `predictions.json`
   (class, confidence, box, severity). Samples are in `results/predictions/`.

```
pip install -r requirements.txt
python scripts/download_india.py && python scripts/prepare_data.py
python -c "from rdd.severity import calibrate; calibrate('data/yolo/labels/train')"   # writes configs/severity_calibration.json
python scripts/train.py --epochs 15 --imgsz 512        # full run (~6 h on CPU)
python scripts/evaluate.py --weights weights/best.pt --imgsz 416
```

## Severity estimation (assumption – RDD2022 has no severity labels)
`extent` = box area / image area (potholes, alligator cracks) or box diagonal / image diagonal (thin longitudinal / transverse cracks).
`score = 0.7 * percentile(extent within same class, training set) + 0.3 * class_weight`
with class weights pothole 1.0, alligator 0.8, transverse 0.5, longitudinal 0.4.
Low < 0.45 <= Medium < 0.70 <= High. Thresholds/weights are heuristics and should be validated with domain experts or severity-labelled data.

## Results (held-out test split, 771 images) — **preliminary**
The model was trained on a CPU-only laptop under a time limit: 2 epochs at 512 px on the full train set, then 3 epochs at 416 px on half of it
(5 epochs total; the planned 15-epoch run was interrupted). It is therefore **under-trained** and the numbers are low.

| Class | Precision | Recall | AP@0.5 |
|---|---|---|---|
| Longitudinal crack | 0.13 | 0.07 | 0.05 |
| Transverse crack | 0.00 | 0.00 | 0.00 |
| Alligator crack | 0.20 | 0.35 | 0.20 |
| Pothole | 0.27 | 0.18 | 0.13 |
| **All** | **0.15** | **0.15** | **mAP50 0.093 / mAP50-95 0.031** |

## Limitations / next steps
- Under-trained: run `scripts/train.py --epochs 50+` on a GPU at 640 px; expect a large jump.
- Transverse cracks are very rare in the India subset (53 train / 6 test boxes), so that class is effectively not learned. Use class-weighting or oversampling.
- Predicted severity is skewed to "High" because the weak model tends to output large alligator-crack boxes; severity quality depends on detection quality.
- Severity is heuristic (no ground truth); it is not a substitute for a pavement-engineering assessment (e.g. depth for potholes).
