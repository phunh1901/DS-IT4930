from ultralytics import YOLO
import cv2

model = YOLO(r"model/OEP_YOLOv11n.pt")

cap = cv2.VideoCapture(0)  # đổi 1,2 nếu cần
if not cap.isOpened():
    raise SystemExit("Không mở được camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    res = model(frame, conf=0.4, verbose=False)[0]
    annotated = res.plot()
    cv2.imshow("YOLOv11 AutoOEP", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()