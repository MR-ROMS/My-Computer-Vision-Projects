import cv2
from ultralytics import YOLO   # ← THIS IS REQUIRED
import time

model = YOLO(MODEL_PATH)

print("Loaded model from:")
print(MODEL_PATH)

print("Classes:")
print(model.names)
results = model(frame, conf=0.01, imgsz=640, verbose=False)[0]