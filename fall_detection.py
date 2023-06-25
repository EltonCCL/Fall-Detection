import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
from ultralytics.yolo.utils.plotting import Annotator
import numpy as np

SOURCE = 'video/video3.mp4'
BED_POLYGON = []

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 live')
    parser.add_argument('--webcam-resolution', default=[640,640], nargs=2, type=int)
    parser.add_argument('--bed-location', default=[0,0,0,0], nargs=2, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    BED_POLYGON.append(args.bed_location)
    # print(frame_height, frame_width)

    # cap = cv2.VideoCapture(1) # Camera Mode
    cap = cv2.VideoCapture(SOURCE) # Video Mode
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


    model = YOLO('yolov8n-pose.pt')

    # box_annotator = sv.BoxAnnotator(
    #     thickness=2,
    #     text_thickness=2,
    #     text_scale=1
    # )
    video = cv2.VideoCapture(SOURCE)
   
    # We need to check if camera
    # is opened previously or not
    if (video.isOpened() == False): 
        print("Error reading video file")
    
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    
    size = (frame_width, frame_height)
    
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    video_result = cv2.VideoWriter('filename1.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)
    video.release()


    while True:
        ret, frame = cap.read()
        if ret == True: 
            results = model.track(frame, verbose=False, conf=0.5, tracker="botsort.yaml", stream=False)[0] # agnostic_nms=True to prevent double detections
            annotator = Annotator(frame, font_size=2, line_width=2)

            # for bed in BED_POLYGON:

            annotator.box_label([  8., 423., 492., 841.], 'BED', color=(102, 255, 102), txt_color=(0,0,0))
            for i, r in enumerate(results):
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    # tag_name = model.names[int(c)] + str(i) + ' ' + str(r.boxes[0].conf.data[0].item())[0:4]
                    tag_name = model.names[int(c)] + str(i) + ' ' + str(r.boxes[0].conf.data[0].item())[0:4]
                    annotator.box_label(b, tag_name, color=(102, 255, 102), txt_color=(0,0,0))

                for keypoint_indx, keypoint in enumerate(r.keypoints.xy[0]):
                    cv2.putText(frame, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame = annotator.result()  
            video_result.write(frame)
            cv2.imshow('YOLO V8 Detection', frame)   

            if (cv2.waitKey(30) == 27):
                break
        else:
            break

    video_result.release()
    cv2.destroyAllWindows()
    print("The video was successfully saved")
if __name__ == "__main__":
    main()