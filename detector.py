from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator


model_path = 'yolov8n-pose.pt'

image_path = 'bus.jpg'
img = cv2.imread(image_path)

model = YOLO(model_path)
results = model(image_path)[0]

for i, r in enumerate(results):    
    annotator = Annotator(img, font_size=2, line_width=2)
    boxes = r.boxes

    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        c = box.cls
        tag_name = model.names[int(c)] + str(i) + ' ' + str(r.boxes[0].conf.data[0].item())[0:4]
        annotator.box_label(b, tag_name, color=(102, 255, 102), txt_color=(0,0,0))

    for keypoint_indx, keypoint in enumerate(r.keypoints.xy[0]):
        cv2.putText(img, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
frame = annotator.result()  
cv2.imshow('YOLO V8 Detection', frame)   

# results = model(image_path)[0]

# for result in results:
#     for keypoint_indx, keypoint in enumerate(result.keypoints.xy[0]):
#         cv2.putText(img, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# cv2.imshow('img', img)
cv2.waitKey(0)