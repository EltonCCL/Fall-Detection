import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
from ultralytics.yolo.utils.plotting import Annotator
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 live')
    parser.add_argument('--webcam-resolution', default=[640,640], nargs=2, type=int)
    parser.add_argument('--source', default='0', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    # print(frame_height, frame_width)

    # cap = cv2.VideoCapture(0) # Camera Mode
    cap = cv2.VideoCapture(args.source) # Video Mode
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    cv2.namedWindow('YOLO V8 Detection', 0)
    cv2.resizeWindow('YOLO V8 Detection', 650, 650)

    model = YOLO('yolov8n-bed.pt')

    while True:
        ret, frame = cap.read()

        results = model(frame, verbose=False, conf=0.5)[0] # agnostic_nms=True to prevent double detections
        annotator = Annotator(frame, font_size=2, line_width=2)

        # for bed in BED_POLYGON:

        # annotator.box_label([100,100,200,200], 'BED', color=(102, 255, 102), txt_color=(0,0,0))

        for i, r in enumerate(results):
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                print(b)
                tag_name = model.names[int(c)] + str(i) + ' ' + str(r.boxes[0].conf.data[0].item())[0:4]
                annotator.box_label(b, tag_name, color=(102, 255, 102), txt_color=(0,0,0))

        frame = annotator.result()  
        cv2.imshow('YOLO V8 Detection', frame)   

        if (cv2.waitKey(30) == 27):
            break
if __name__ == "__main__":
    main()