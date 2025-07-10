# 🔥 Thermal-Attention-YOLOv8

This project focuses on enhancing the performance of YOLOv8 in thermal image domains by integrating attention mechanisms into the model architecture.

Traditional object detection models are typically trained and optimized on RGB image datasets, which may not generalize well to thermal imagery due to its distinct characteristics and limited texture details. To address this, we explore and embed various attention modules into the YOLOv8 architecture, aiming to guide the network to focus on informative thermal regions and improve detection performance — particularly for small and low-contrast objects.

As a baseline, the FLIR thermal dataset is used for evaluation and benchmarking.

---

## ✅ Attention Modules Checklist

- ✅ **SKAttention** (implemented and tested)
- ⏳ **CBAM** (coming soon)
- ✅ **PSA** (implemented and tested)
- ✅ **SimAM** (implemented and tested)
- ⏳ **GAM** (planned)
- ⏳ **SE** (planned)

---

## ⚙️ Example Configuration Block with SKNet

```yaml
- [-1, 1, SKAttention, [1024, [3, 5, 7], 16]]
```

- `1024` — Number of input/output channels  
- `[3, 5, 7]` — Multi-scale kernel sizes for spatial attention  
- `16` — Channel reduction ratio in the attention block  

---
## ⚙️ Example Configuration Block with PSA
```yaml
- [-1, 1, PSAPlug, [1024,4]]
```

- `1024` — Number of input/output channels  
- `[3, 5, 7]` — Multi-scale kernel sizes for spatial attention  
- `16` — Channel reduction ratio in the attention block  

---

## 🧪 Sample Training Script

```python
from ultralytics import YOLO

model = YOLO("yolov8s-sk.yaml")
model.train(
    data="flir-dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    name="yolov8s-sk"
)
```

---

## 📦 Dataset

This project uses the [FLIR Thermal Dataset](https://www.flir.com/oem/adas/adas-dataset-form/), which includes annotated thermal images designed for ADAS (Advanced Driver-Assistance Systems) research.

---

## 📌 Future Plans

- Benchmark each attention module on the same dataset.
- Analyze impact of each kernel in SKAttention during training.
- Extend to RGB-T fusion models for multi-modal detection.
- Add performance comparison table (mAP, precision, recall) across modules.
