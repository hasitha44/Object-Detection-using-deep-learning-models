import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
video_path = r'C:\Users\rajab\OneDrive\Videos\SampleVideo_1280x720_1mb.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
output_path = 'detected_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    predictions = results.pandas().xyxy[0]
    for index, row in predictions.iterrows():
        x1, y1, x2, y2, confidence, class_id, class_name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class']), row['name']
        label = f'{class_name} {confidence:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    out.write(frame)
cap.release()
out.release()
print(f"Detected objects in video saved to {output_path}")