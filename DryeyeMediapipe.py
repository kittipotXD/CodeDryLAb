import cv2
import numpy as np
import math
from ultralytics import YOLO
import mediapipe as mp

# Load YOLO model
model = YOLO(r'C:\Users\lnwTutor\Desktop\DRYEYE D\best8xmAP93.pt')

# Define class mapping
class_mapping = {
    0: "ตาแห้ง",
    1: "ตาแห้ง",
    2: "ตาปกติ"
}

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

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

        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe to get landmarks
        result = face_mesh.process(rgb_frame)

        # If landmarks are detected, calculate eye heights
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                right_eye_top_index = 159  # Top edge of right eye
                right_eye_bottom_index = 145  # Bottom edge of right eye
                left_eye_top_index = 386  # Top edge of left eye
                left_eye_bottom_index = 374  # Bottom edge of left eye

                # Calculate eye coordinates
                right_eye_top = (int(face_landmarks.landmark[right_eye_top_index].x * frame.shape[1]),
                                 int(face_landmarks.landmark[right_eye_top_index].y * frame.shape[0]))
                right_eye_bottom = (int(face_landmarks.landmark[right_eye_bottom_index].x * frame.shape[1]),
                                    int(face_landmarks.landmark[right_eye_bottom_index].y * frame.shape[0]))
                left_eye_top = (int(face_landmarks.landmark[left_eye_top_index].x * frame.shape[1]),
                                int(face_landmarks.landmark[left_eye_top_index].y * frame.shape[0]))
                left_eye_bottom = (int(face_landmarks.landmark[left_eye_bottom_index].x * frame.shape[1]),
                                   int(face_landmarks.landmark[left_eye_bottom_index].y * frame.shape[0]))

                # Calculate eye heights
                right_eye_height_pixels = math.dist(right_eye_top, right_eye_bottom)
                left_eye_height_pixels = math.dist(left_eye_top, left_eye_bottom)

                # Convert heights from pixels to mm
                right_eye_height_mm = right_eye_height_pixels * pixel_to_mm_ratio
                left_eye_height_mm = left_eye_height_pixels * pixel_to_mm_ratio

                # Store eye heights
                right_eye_heights.append(right_eye_height_mm)
                left_eye_heights.append(left_eye_height_mm)

                # Draw eye edges
                cv2.line(frame, right_eye_top, right_eye_bottom, (0, 255, 0), 2)
                cv2.line(frame, left_eye_top, left_eye_bottom, (0, 255, 0), 2)

                # Display measurements on frame
                cv2.putText(frame, f"Right Eye Height: {right_eye_height_mm:.2f} mm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Left Eye Height: {left_eye_height_mm:.2f} mm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Increment capture count
        capture_count += 1

        # Show the captured frame with detection results
        cv2.imshow("Captured Image", frame)

        # Check if maximum captures reached
        if capture_count == num_captures:
            print("Reached maximum captures. Analysis complete.")

            # Calculate average heights
            avg_right_eye_height = sum(right_eye_heights) / len(right_eye_heights)
            avg_left_eye_height = sum(left_eye_heights) / len(left_eye_heights)

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
                        print("ตาของคุณอยู่ในเกณฑ์ดี โปรดระวังการจ้องหน้าจอมากเกินไป")
                    elif 25 < avg_EYE <= 50:
                        print("ตาของคุณอยู่ในเกณฑ์ปกติ ต้องกระพริบตาบ่อยขึ้นเพื่อป้องกันโรคตาแห้ง")
                    else:
                        print("โปรดทำแบบสอบเพิ่มเติมเพื่อป้องกันโรคตาแห้ง")
                elif cls == "ตาแห้ง":
                    if 1 < avg_EYE <= 100:
                        print("ตาของคุณเป็นตาแห้ง ทางเรากำลังส่งข้อมูลไปยังโรงพยาบาลเพื่อประสานงานในการรักษา")

            # แสดงผลการวัดความสูงของตา
            print("\nผลการวัดความสูงของตา:")
            print(f"ค่าเฉลี่ยความสูงของตาขวา: {avg_right_eye_height:.2f} mm")
            print(f"ค่าเฉลี่ยความสูงของตาซ้าย: {avg_left_eye_height:.2f} mm")

            # Clear detections and reset capture count for next round
            detections.clear()
            right_eye_heights.clear()
            left_eye_heights.clear()
            capture_count = 0

    elif key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
