import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt') # select your model.pt path
    model.predict(source='ultralytics/assets/bus.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  classes=0,
                )