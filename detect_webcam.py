import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    predictions = results.pandas().xyxy[0]
    for index, row in predictions.iterrows():
        x1, y1, x2, y2, confidence, class_id, class_name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class']), row['name']
        label = f'{class_name} {confidence:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()