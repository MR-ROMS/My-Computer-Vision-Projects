from ultralytics import YOLO

model = YOLO(r'D:\CV_Projects\runs\detect\runs\train\traffic_v6\weights\best.pt')
results = model.predict(
    source  = r'Videos/traffic4.mp4',
    conf    = 0.1,
    imgsz   = 640,
    device  = 0,
    show    = False,
    save    = False,
    verbose = True,
)

for r in results[:5]:
    print(r.boxes)