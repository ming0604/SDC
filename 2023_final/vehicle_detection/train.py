from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.yaml')  # build a model from YAML 

# Train the model
model.train(data='data.yaml', epochs=1000, batch=-1, imgsz=640, device=[0, 1])
