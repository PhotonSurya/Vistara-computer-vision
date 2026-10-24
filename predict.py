
from ultralytics import YOLO
import cv2
import json

# ---- CONFIG ----
MODEL_PATH = "best.pt"
INPUT_IMAGE = "test.webp"
OUTPUT_IMAGE = "output.jpg"
OUTPUT_JSON = "output.json"

# Load the model
model = YOLO(MODEL_PATH)

# Load image
img = cv2.imread(INPUT_IMAGE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {INPUT_IMAGE}")

# Run prediction
results = model(img, conf=0.25)[0]

# Store detection results
detections = []

# Draw boxes and collect JSON data
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf = float(box.conf[0])
    class_id = int(box.cls[0])
    class_name = results.names[class_id]

    detections.append({
        "class_id": class_id,
        "class": class_name,
        "confidence": round(conf, 4),
        "bounding_box": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
    })

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Label
    label = f"{class_name} {conf:.2f}"
    cv2.putText(
        img, label, (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )

# Save annotated image
cv2.imwrite(OUTPUT_IMAGE, img)

# Create JSON output
output_data = {
    "image": INPUT_IMAGE,
    "total_detections": len(detections),
    "detections": detections
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Detection complete! Image saved as: {OUTPUT_IMAGE}")
print(f"JSON results saved as: {OUTPUT_JSON}")
print(f"Total detections: {len(detections)}")

