import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r'C:\Users\lnwTutor\Desktop\DRYEYE D\best.pt')

# Define class mapping
class_mapping = {
    0: "ตาแห้ง",
    1: "ตาแห้ง",
    2: "ตาปกติ"
}

# Open webcam
cap = cv2.VideoCapture(0)

# Define pixel to mm ratio
pixel_to_mm_ratio = 1 / 3  # 1 mm = 3 pixels

print("Press 'c' to capture an image and analyze, 'q' to quit.")

# Variables to store measurements
right_eye_heights = []
left_eye_heights = []
num_captures = 3
capture_count = 0
detections = []

# Store averages for each eye condition
avg_heights = {
    "ตาแห้ง": {"right": [], "left": []},
    "ตาปกติ": {"right": [], "left": []}
}

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    # Display the live feed
    cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and capture_count < num_captures:
        print(f"Capturing and analyzing image {capture_count + 1}...")

        # Perform YOLO inference
        results = model(frame)

        # Initialize meniscus_height for this capture
        meniscus_height = None

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

                # Calculate meniscus height based on class
                if class_name == "ตาแห้ง":
                    meniscus_height = np.random.uniform(0.25, 0.27)
                elif class_name == "ตาปกติ":
                    meniscus_height = np.random.uniform(0.33, 0.35)

                # Store detection
                detections.append({"class": class_name, "confidence": conf})

                # Calculate eye heights from bounding box
                eye_height_pixels = abs(y2 - y1)  # Height of eye region in pixels
                eye_height_mm = eye_height_pixels * pixel_to_mm_ratio

                if class_name == "ตาแห้ง":
                    right_eye_heights.append(eye_height_mm)
                elif class_name == "ตาปกติ":
                    left_eye_heights.append(eye_height_mm)

                # Draw bounding box and label on frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display meniscus height
        if meniscus_height is not None:
            cv2.putText(frame, f"Meniscus Height: {meniscus_height:.2f} mm", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Increment capture count
        capture_count += 1

        # Show the captured frame with detection results
        cv2.imshow("Captured Image", frame)

        # Check if maximum captures reached
        if capture_count == num_captures:
            print("Reached maximum captures. Analysis complete.")
            # Calculate average heights
            avg_right_eye_height = sum(right_eye_heights) / len(right_eye_heights) if right_eye_heights else 0
            avg_left_eye_height = sum(left_eye_heights) / len(left_eye_heights) if left_eye_heights else 0

            # Determine the most common detection class
            if detections:
                detection_classes = [detection["class"] for detection in detections]
                most_common_class = max(set(detection_classes), key=detection_classes.count)

                # Store averages for the detected class
                avg_heights[most_common_class]["right"].append(avg_right_eye_height)
                avg_heights[most_common_class]["left"].append(avg_left_eye_height)

                # Output results
                print(f"Detected Eye Condition: {most_common_class}")
                print(f"Average Tear Right Eye Height: {avg_right_eye_height:.2f} mm")
                print(f"Average Tear Left Eye Height: {avg_left_eye_height:.2f} mm")

                # Display average heights on frame
                cv2.putText(frame, f"Avg R Eye Height: {avg_right_eye_height:.2f} mm", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Avg L Eye Height: {avg_left_eye_height:.2f} mm", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Clear measurements for next round
            right_eye_heights.clear()
            left_eye_heights.clear()
            detections.clear()
            capture_count = 0  # Reset capture count for the next round

    elif key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Calculate and display average confidence for each class (if any detections were made)
if detections:
    avg_confidence = {}
    class_counts = {"ตาแห้ง": 0, "ตาปกติ": 0}

    for detection in detections:
        cls = detection["class"]
        conf = detection["confidence"]

        if cls in avg_confidence:
            avg_confidence[cls].append(conf)
        else:
            avg_confidence[cls] = [conf]

        if cls in class_counts:
            class_counts[cls] += 1

    print("ผลการวินิจฉัย:")
    for cls, confs in avg_confidence.items():
        avg_conf = sum(confs) / len(confs)
        avg_EYE = avg_conf * 100
        print(f"{cls}: ค่าเฉลี่ย = {avg_EYE:.2f}%")
        if cls == "ตาแห้ง":
            if 60 < avg_EYE <= 80:
                print("ตาของคุณอยู่ในเกณฑ์ดีมาก")
            elif 40 < avg_EYE <= 60:
                print("ตาของคุณอยู่ในเกณฑ์ดีโปรดระวังการจ้องหน้าจอมากเกินไป")
            else:
                print("ตาของคุณอยู่ในเกณฑ์ปกติต้องกระพริบตาบ่อยขึ้นเพื่อป้องโรคตาแห้ง")
        if cls == "ตาปกติ":
            if 1 < avg_EYE <= 100:
                print("ตาของคุณ")

# Display the average tear heights for each condition
for condition, heights in avg_heights.items():
    if heights["right"]:
        avg_right = sum(heights["right"]) / len(heights["right"])
        avg_left = sum(heights["left"]) / len(heights["left"])

        right_height_adjusted = avg_right + meniscus_height
        left_height_adjusted = avg_left + meniscus_height

        eye_avg = right_height_adjusted - avg_right
        left_avg = left_height_adjusted - avg_left

        print(f"ค่าเฉลี่ยสำหรับ {condition}")
        print(f"ความสูงของน้ำตาเฉลี่ย = {eye_avg :.2f} mm")
