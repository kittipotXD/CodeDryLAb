import cv2
import numpy as np
import math
from ultralytics import YOLO
import mediapipe as mp

# โหลดโมเดล YOLO
model = YOLO(r'C:\Users\lnwTutor\Desktop\DRYEYE D\best8xmAP93.pt')

# กำหนดการแมปคลาส
class_mapping = {
    0: "ตาแห้ง",
    1: "ตาแห้ง",
    2: "ตาปกติ"
}

# เริ่มต้น MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)

# กำหนดอัตราส่วนพิกเซลเป็นมิลลิเมตร
pixel_to_mm_ratio = 1 / 3  # 1 มม. = 3 พิกเซล

print("กด 'c' เพื่อจับภาพและวิเคราะห์, 'q' เพื่อออกจากโปรแกรม.")

# ตัวแปรเก็บการวัด
right_eye_heights = []
left_eye_heights = []
num_captures = 3
capture_count = 0
detections = []

while True:
    # จับภาพจากกล้องเว็บแคม
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถจับภาพจากกล้องเว็บแคมได้.")
        break

    # แสดงจำนวนภาพที่จับได้บนเฟรม
    cv2.putText(frame, f"Captured Images: {capture_count}/{num_captures}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # แสดงการถ่ายทอดสด
    cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and capture_count < num_captures:
        print(f"กำลังจับภาพและวิเคราะห์ภาพที่ {capture_count + 1}...")

        # ทำการทำนายผลด้วย YOLO
        results = model(frame)

        # ประมวลผลผลลัพธ์จาก YOLO
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # แมปคลาส
                class_name = class_mapping.get(cls, "Unknown")
                print(f"ตรวจจับคลาส: {class_name} ด้วยความมั่นใจ {conf:.2f}")

                # เก็บการตรวจจับ
                detections.append({"class": class_name, "confidence": conf})

                # วาดกรอบสี่เหลี่ยมและป้ายชื่อบนเฟรม
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # แปลงเฟรมเป็น RGB สำหรับการประมวลผล MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ประมวลผลเฟรมด้วย MediaPipe เพื่อรับตำแหน่งของจุด landmarks
        result = face_mesh.process(rgb_frame)

        # หากมีการตรวจพบ landmarks, คำนวณความสูงของตา
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                right_eye_top_index = 159  # ขอบด้านบนของตาขวา
                right_eye_bottom_index = 145  # ขอบด้านล่างของตาขวา
                left_eye_top_index = 386  # ขอบด้านบนของตาซ้าย
                left_eye_bottom_index = 374  # ขอบด้านล่างของตาซ้าย

                # คำนวณพิกัดของตา
                right_eye_top = (int(face_landmarks.landmark[right_eye_top_index].x * frame.shape[1]),
                                 int(face_landmarks.landmark[right_eye_top_index].y * frame.shape[0]))
                right_eye_bottom = (int(face_landmarks.landmark[right_eye_bottom_index].x * frame.shape[1]),
                                    int(face_landmarks.landmark[right_eye_bottom_index].y * frame.shape[0]))
                left_eye_top = (int(face_landmarks.landmark[left_eye_top_index].x * frame.shape[1]),
                                int(face_landmarks.landmark[left_eye_top_index].y * frame.shape[0]))
                left_eye_bottom = (int(face_landmarks.landmark[left_eye_bottom_index].x * frame.shape[1]),
                                   int(face_landmarks.landmark[left_eye_bottom_index].y * frame.shape[0]))

                # คำนวณความสูงของตา
                right_eye_height_pixels = math.dist(right_eye_top, right_eye_bottom)
                left_eye_height_pixels = math.dist(left_eye_top, left_eye_bottom)

                # แปลงความสูงจากพิกเซลเป็นมิลลิเมตร
                right_eye_height_mm = right_eye_height_pixels * pixel_to_mm_ratio
                left_eye_height_mm = left_eye_height_pixels * pixel_to_mm_ratio

                # เก็บความสูงของตา
                right_eye_heights.append(right_eye_height_mm)
                left_eye_heights.append(left_eye_height_mm)

                # วาดเส้นขอบของตา
                cv2.line(frame, right_eye_top, right_eye_bottom, (0, 255, 0), 2)
                cv2.line(frame, left_eye_top, left_eye_bottom, (0, 255, 0), 2)

                # แสดงการวัดบนเฟรม
                cv2.putText(frame, f"Right Eye Height: {right_eye_height_mm:.2f} mm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Left Eye Height: {left_eye_height_mm:.2f} mm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # เพิ่มจำนวนครั้งที่จับภาพ
        capture_count += 1

        # แสดงภาพที่จับได้พร้อมผลการตรวจจับ
        cv2.imshow("Captured Image", frame)

        # ตรวจสอบว่าจับภาพครบตามจำนวนที่กำหนดหรือไม่
        if capture_count == num_captures:
            print("จับภาพครบตามจำนวนแล้ว การวิเคราะห์เสร็จสมบูรณ์.")

            # คำนวณค่าเฉลี่ยของความสูง
            avg_right_eye_height = sum(right_eye_heights) / len(right_eye_heights)
            avg_left_eye_height = sum(left_eye_heights) / len(left_eye_heights)

            # คำนวณค่าเฉลี่ยของความมั่นใจสำหรับแต่ละคลาส
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

            # เคลียร์การตรวจจับและรีเซ็ตการจับภาพสำหรับรอบถัดไป
            detections.clear()
            right_eye_heights.clear()
            left_eye_heights.clear()
            capture_count = 0

    elif key == ord('q'):
        break

# ปล่อยการจับภาพและปิดหน้าต่าง
cap.release()
cv2.destroyAllWindows()
