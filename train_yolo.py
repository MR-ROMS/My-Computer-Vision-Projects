import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from ultralytics import YOLO
import torch

DATA_YAML  = r'D:\CV_Projects\Trafik_Bitirme\dataset\data.yaml'
MODEL_SIZE = 'yolov8m.pt'
EPOCHS     = 100
IMG_SIZE   = 960
BATCH_SIZE = 4
PROJECT    = 'runs/train'
RUN_NAME   = 'traffic_v9'

def main():
    torch.cuda.empty_cache()
    model = YOLO(MODEL_SIZE)

    results = model.train(
        data          = DATA_YAML,
        epochs        = EPOCHS,
        imgsz         = IMG_SIZE,
        batch         = BATCH_SIZE,
        nbs           = 16,
        workers       = 0,
        amp           = True,
        cache         = 'disk',
        project       = PROJECT,
        name          = RUN_NAME,
        exist_ok      = True,
        device        = 0,
        # Augmentation
        hsv_h         = 0.015,
        hsv_s         = 0.5,
        hsv_v         = 0.4,
        degrees       = 10.0,
        scale         = 0.5,
        mosaic        = 1.0,
        close_mosaic  = 10,
        mixup         = 0.0,
        copy_paste    = 0.0,
        fliplr        = 0.0,
        flipud        = 0.0,
        # Loss
        box           = 7.5,
        cls           = 1.5,
        # Optimization
        lr0           = 0.01,
        lrf           = 0.01,
        cos_lr        = True,
        warmup_epochs = 5,
        patience      = 50,
        deterministic = True,
        seed          = 42,
        # Saving
        save          = True,
        save_period   = 5,
        plots         = True,
        verbose       = True,
    )

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print("=" * 50)

if __name__ == '__main__':
    main()