import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r'C:\Users\lnwTutor\Desktop\DRYEYE D\best8xmAP93.pt')

# Define class mapping
class_mapping = {
    0: "ตาแห้ง",
    1: "ตาแห้ง",
    2: "ตาปกติ"
}

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'c' to capture an image and analyze, 'q' to quit.")

num_captures = 3
capture_count = 0
detections = []

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    # Display capture count on the frame
    cv2.putText(frame, f"Captured Images: {capture_count}/{num_captures}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the live feed
    cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and capture_count < num_captures:
        print(f"Capturing and analyzing image {capture_count + 1}...")

        # Perform YOLO inference
        results = model(frame)

        # Process YOLO results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Map class
                class_name = class_mapping.get(cls, "Unknown")
                print(f"Detected class: {class_name} with confidence {conf:.2f}")

                # Store detection
                detections.append({"class": class_name, "confidence": conf})

                # Draw bounding box and label on frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Increment capture count
        capture_count += 1

        # Show the captured frame with detection results
        cv2.imshow("Captured Image", frame)

        # Check if maximum captures reached
        if capture_count == num_captures:
            print("Reached maximum captures. Analysis complete.")

            # Calculate average confidence for each class
            avg_confidence = {}
            for detection in detections:
                cls = detection["class"]
                conf = detection["confidence"]

                if cls in avg_confidence:
                    avg_confidence[cls].append(conf)
                else:
                    avg_confidence[cls] = [conf]

            print("ผลการวินิจฉัย:")
            for cls, confs in avg_confidence.items():
                avg_conf = sum(confs) / len(confs)
                avg_EYE = avg_conf * 100
                print(f"{cls}: ค่าเฉลี่ย = {avg_EYE:.2f}%")
                if cls == "ตาปกติ":
                    if 75 < avg_EYE <= 100:
                        print("ตาของคุณอยู่ในเกณฑ์ดีมาก")
                    elif 50 < avg_EYE <= 75:
                        print("ตาของคุณอยู่ในเกณฑ์ดีโปรดระวังการจ้องหน้าจอมากเกินไป")
                    elif 25 < avg_EYE <= 50:
                        print("ตาของคุณอยู่ในเกณฑ์ปกติต้องกระพริบตาบ่อยขึ้นเพื่อป้องโรคตาแห้ง")
                    else:
                        print("โปรดทำแบบสอบเพิ่มเติมเพื่อป้องกันโรคตาแห้ง")
                elif cls == "ตาแห้ง":
                    if 1 < avg_EYE <= 100:
                        print("ตาของคุณเป็นตาแห้ง ทางเรากำลังส่งข้อมูลไปยังโรงพยาบาลเพื่อประสานงานในการรักษา")

            # Clear detections and reset capture count for next round
            detections.clear()
            capture_count = 0

    elif key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
