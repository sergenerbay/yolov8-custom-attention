
from ultralytics import YOLO
# Önceden eğitilmiş ağırlıklarla modeli yükle
model = YOLO("yolov8s-PSA.yaml")
model.train()

