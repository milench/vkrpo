from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # model loading

results = model.train(
    data="/Users/milenabuzyleva/Desktop/computerVision/boat recognition/data.yaml",
    imgsz=640,
    epochs=2,
    batch=16,
    name='boat_detect',
    save=True)

