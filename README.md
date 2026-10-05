# Construction-Site Safety — YOLO PPE Detection

> A custom-trained YOLO model that detects personal protective equipment (PPE) compliance on construction sites in real time: **helmet / no-helmet / vest / no-vest / person**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111F68)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-6706CE)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Workers without a helmet or high-visibility vest are a leading cause of preventable site injuries. This project trains a YOLO detector on a labelled construction-safety dataset and runs it on webcam or video streams to flag PPE violations.

| Class | Meaning |
|---|---|
| `helmet` | Worker wearing a hard hat ✅ |
| `no-helmet` | Head without a hard hat ⚠️ |
| `vest` | High-visibility vest worn ✅ |
| `no-vest` | Missing vest ⚠️ |
| `person` | Any person in frame |

## Pipeline

1. **Data**: construction-safety dataset exported from Roboflow in YOLO format (images + bounding-box labels + `data.yaml`)
2. **Training** (`notebooks/01_train_construction_safety_yolo.ipynb`): Ultralytics YOLO on Google Colab GPU, transfer learning from pretrained YOLOv8n / YOLO11s / YOLO12s weights (30 epochs), plus validation and prediction samples
3. **Export**: best checkpoint saved as `models/best_2025.pt`
4. **Real-time inference** (`notebooks/02_realtime_inference.ipynb`): OpenCV webcam/video loop with bounding boxes, class labels, confidence, and FPS overlay

## Quick start

```bash
git clone https://github.com/ZakariaHibaoui2/yolo-object-detection.git
cd yolo-object-detection
pip install -r requirements.txt
```

**Detect on webcam with the trained model**

```bash
yolo predict model=models/best_2025.pt source=0 show=True
```

**Detect on an image or video**

```python
from ultralytics import YOLO
model = YOLO("models/best_2025.pt")
results = model.predict("site_photo.jpg", conf=0.5, save=True)
```

**Retrain**: open notebook 01 in Colab, add your Roboflow key as `YOUR_ROBOFLOW_API_KEY`, and run all cells.

## Project structure

```
├── models/best_2025.pt                          # Trained PPE detector (5 classes)
├── notebooks/
│   ├── 01_train_construction_safety_yolo.ipynb  # Dataset download, training, validation
│   └── 02_realtime_inference.ipynb              # Webcam / video inference with OpenCV
├── docs/Guide_to_Object_Detection.docx          # Write-up: concepts and methodology
└── requirements.txt
```

## Tech stack

Python · Ultralytics YOLO (v8 / 11 / 12) · PyTorch · OpenCV · Roboflow · Google Colab

## Author

**Zakaria Hibaoui** — [GitHub](https://github.com/ZakariaHibaoui2) · [LinkedIn](https://www.linkedin.com/in/zakaria-hibaoui-08b235337/)

## License

[MIT](LICENSE)
