from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolov8s.yaml')  # build a model from YAML 
    # Train the model
    model.train(data='data.yaml', epochs=1000, batch=-1, imgsz=480, device=0, amp=False)

if __name__ == '__main__':
   main()
