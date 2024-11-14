import cv2
import numpy as np
import os
from ultralytics import YOLO

# กำหนดโฟลเดอร์ที่บันทึกภาพ (ในที่นี้จะเป็นโฟลเดอร์ปัจจุบัน)
output_folder = "."

# ลบไฟล์ภาพที่บันทึกไว้ก่อนหน้านี้ในโฟลเดอร์
for filename in os.listdir(output_folder):
    if filename.endswith("_last_capture.jpg"):
        file_path = os.path.join(output_folder, filename)
        try:
            os.remove(file_path)
            print(f"ลบไฟล์เก่า: {filename}")
        except Exception as e:
            print(f"ไม่สามารถลบไฟล์ {filename}: {e}")

# โหลดโมเดล YOLO เพียงครั้งเดียวก่อนเปิดกล้อง
try:
    model = YOLO(r'C:\Users\lnwTutor\Desktop\DRYEYE D\bestSize X mAP0.93.pt')
except Exception as e:
    print("ไม่สามารถโหลดโมเดลได้:", e)
    exit()

# กำหนดการแมปคลาส
class_mapping = {
    0: "ตาแห้ง",
    1: "ตาแห้ง",
    2: "ตาปกติ"
}

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

print("กด 'c' เพื่อจับภาพและวิเคราะห์, 'q' เพื่อออกจากโปรแกรม.")

num_captures = 3
capture_count = 0
detections = []
patient_name = ""
entering_name = True  # สถานะสำหรับการกรอกชื่อ
latest_frame = None  # ตัวแปรเก็บภาพล่าสุดที่ถ่าย

while True:
    # จับภาพจากกล้องเว็บแคม
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถจับภาพจากกล้องได้")
        break

    # ลดขนาดภาพเพื่อให้แสดงผลเร็วขึ้น
    frame = cv2.resize(frame, (640, 480))

    # ตรวจสอบสถานะว่ากำลังกรอกชื่อหรือไม่
    if entering_name:
        cv2.putText(frame, "Enter patient name: " + patient_name, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(0) & 0xFF
        if key == 8:  # Backspace เพื่อลบตัวอักษร
            patient_name = patient_name[:-1]
        elif key == 13:  # Enter เพื่อยืนยันชื่อ
            entering_name = False
        elif key == ord('q'):
            break
        elif key != 255:  # ตรวจสอบว่าคีย์ไม่ใช่ปุ่มพิเศษ
            patient_name += chr(key)
        continue

    # แสดงจำนวนภาพที่จับได้บนเฟรม
    cv2.putText(frame, f"Captured Images: {capture_count}/{num_captures}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Patient: {patient_name}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # แสดงการถ่ายทอดสด
    cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", frame)
    key = cv2.waitKey(5) & 0xFF

    if key == ord('c') and capture_count < num_captures:
        print(f"กำลังจับภาพและวิเคราะห์ภาพที่ {capture_count + 1}...")
        latest_frame = frame.copy()  # เก็บเฟรมล่าสุดที่ถ่าย

        # ทำการทำนายผลด้วย YOLO
        results = model(latest_frame)

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
                cv2.rectangle(latest_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(latest_frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # เพิ่มจำนวนครั้งที่จับภาพ
        capture_count += 1

        # แสดงภาพที่จับได้พร้อมผลการตรวจจับ
        cv2.imshow("Captured Image", latest_frame)

        # ตรวจสอบว่าจับภาพครบตามจำนวนที่กำหนดหรือไม่
        if capture_count == num_captures:
            print("จับภาพครบตามจำนวนแล้ว การวิเคราะห์เสร็จสมบูรณ์.")

            # บันทึกภาพสุดท้ายในรอบนี้
            if latest_frame is not None:
                cv2.imwrite(f"{patient_name}_last_capture.jpg", latest_frame)
                print(f"บันทึกภาพสุดท้ายเป็นไฟล์: {patient_name}_last_capture.jpg")

            # คำนวณค่าเฉลี่ยของความมั่นใจสำหรับแต่ละคลาส
            avg_confidence = {}
            for detection in detections:
                cls = detection["class"]
                conf = detection["confidence"]

                if cls in avg_confidence:
                    avg_confidence[cls].append(conf)
                else:
                    avg_confidence[cls] = [conf]

            print(f"ผลการวินิจฉัยสำหรับ {patient_name}:")
            for cls, confs in avg_confidence.items():
                avg_conf = sum(confs) / len(confs)
                avg_EYE = avg_conf * 100
                print(f"{cls}: ค่าเฉลี่ย = {avg_EYE:.2f}%")
                if cls == "ตาปกติ":
                    if 75 < avg_EYE <= 100:
                        print(f"ตาของ {patient_name} อยู่ในเกณฑ์ดีมาก")
                    elif 50 < avg_EYE <= 75:
                        print(f"ตาของ {patient_name} อยู่ในเกณฑ์ดี โปรดระวังการจ้องหน้าจอมากเกินไป")
                    elif 25 < avg_EYE <= 50:
                        print(f"ตาของ {patient_name} อยู่ในเกณฑ์ปกติ ต้องกระพริบตาบ่อยขึ้นเพื่อป้องกันโรคตาแห้ง")
                    else:
                        print(f"โปรดทำแบบสอบเพิ่มเติมเพื่อป้องกันโรคตาแห้ง")
                elif cls == "ตาแห้ง":
                    if 1 < avg_EYE <= 100:
                        print(f"ตาของ {patient_name} เป็นตาแห้ง ทางเรากำลังส่งข้อมูลไปยังโรงพยาบาลเพื่อประสานงานในการรักษา")

            # เคลียร์การตรวจจับและรีเซ็ตจำนวนจับภาพสำหรับการรอบถัดไป
            detections.clear()
            capture_count = 0
            entering_name = True  # กลับไปกรอกชื่อคนไข้ใหม่

    elif key == ord('q'):
        break

# ปล่อยการจับภาพและปิดหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
