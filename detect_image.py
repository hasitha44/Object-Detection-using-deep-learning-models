import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img_path = r'C:\Users\rajab\OneDrive\Pictures\homenature.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lp
results = model(img)

predictions = results.pandas().xyxy[0]

for index, row in predictions.iterrows():
    x1, y1, x2, y2, confidence, class_id, class_name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class']), row['name']
    label = f'{class_name} {confidence:.2f}'
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('detected_image.jpg', img_bgr)
print(f"Detected objects saved to detected_image.jpg")