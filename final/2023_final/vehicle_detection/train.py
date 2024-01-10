from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolov8n.yaml')  # build a model from YAML 
    # Train the model
    model.train(data='data.yaml', epochs=500, batch=-1, imgsz=640, device=0, amp=False)

if __name__ == '__main__':
   main()
