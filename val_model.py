from ultralytics import YOLO

def main():
    model = YOLO(r'D:\CV_Projects\runs\detect\runs\train\traffic_v7\weights\best.pt')

    model.val(
        data    = r'D:\CV_Projects\Trafik_Bitirme\dataset\data.yaml',
        plots   = True,
        workers = 0,
    )

if __name__ == '__main__':
    main()