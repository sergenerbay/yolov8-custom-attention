
from ultralytics import YOLO
# Önceden eğitilmiş ağırlıklarla modeli yükle
model = YOLO("yolov8-Simam.yaml")
model.train()