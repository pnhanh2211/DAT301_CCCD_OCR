from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")


# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("ccctes1.jpg")