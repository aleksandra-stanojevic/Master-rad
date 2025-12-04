from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8s.pt")

# Train the model
results = model.train(
    data="lung_det.yaml",
    epochs=50,
    imgsz=640,
    batch=2,          
    workers=0,        # IMPORTANT for avoiding dataloader crashes
    device=0,         # force GPU; use "cpu" if needed
    cache=False,      # do NOT cache dataset into RAM
    project="lung_yolo_runs",
    name="lung_detector",
)

# Evaluate the model on validation set
metrics = model.val()
print("Validation metrics:", metrics)

# Test on a few images to visually inspect
model.predict(
    source="test_images/",
    save=True,
    imgsz=640,
    conf=0.25,
)

print("\nTraining complete!")
print("Best model saved at: lung_yolo_runs/yolo_lung_det/weights/best.pt")