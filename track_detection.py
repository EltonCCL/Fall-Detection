# Modified form https://github.com/SkalskiP/yolov8-native-tracking/blob/master/main.py
import cv2
from pathlib import Path
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator
import os

OUTPUT_FOLDER = 'output'
# SOURCE = 'video/raw-los-angeles-streets-episode-2.mp4'

HUMAN_STATUS = {'SAFE': {'Description': 'SAFE', 'Color': (43, 153, 18)},
                'FALL': {'Description': 'FALL', 'Color': (9, 9, 181)},
                'onBED': {'Description': 'onBED', 'Color': (173, 15, 12)}}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='fall-detection')
    parser.add_argument('--bed-location', default=[], nargs='+', type=int)
    parser.add_argument('--source', default='0', type=str)
    parser.add_argument('--model', default='yolov8n-pose.pt', nargs=1, type=str)
    parser.add_argument('--verbose', default=True,action=argparse.BooleanOptionalAction)
    parser.add_argument('--stream', default=True,action=argparse.BooleanOptionalAction)
    parser.add_argument('--show', default=True,action=argparse.BooleanOptionalAction)
    parser.add_argument('--conf', default=0.5, type=float)
    args = parser.parse_args()
    return args


def inside_rectangle(point_xy, rectangle_box: list):
    px1, py1, px2, py2 = rectangle_box
    x, y = point_xy
    if px1 <= x and x <= px2 and py1 <= y and y <= py2:
        return True


def human_status(rectangle_box: list, keypoints, beds):
    for bed in beds:
        # print(keypoints[11], keypoints[12], bed)
        if inside_rectangle(keypoints[11], bed) and inside_rectangle(keypoints[12], bed):
            return 'onBED'
    px1, py1, px2, py2 = rectangle_box
    if (px2 - px1) > (py2 - py1):
        return 'FALL'
    else:
        return 'SAFE'


def main():
    beds = []

    args = parse_arguments()
    model = YOLO(args.model)
    video = cv2.VideoCapture(args.source)

    if args.bed_location != []:
        beds.append(args.bed_location)

    isExist = os.path.exists(OUTPUT_FOLDER)
    if not isExist:
        os.makedirs(OUTPUT_FOLDER)

    if (video.isOpened() == False):
        print("Error reading video file")
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    size = (frame_width, frame_height)
    output_name = OUTPUT_FOLDER + '/' + str(Path(args.source).stem + '.avi')
    video_result = cv2.VideoWriter(output_name,
                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                   fps, size)

    # cv2.namedWindow('yolov8', 0)
    # cv2.resizeWindow('yolov8', 900, 900)
    video.release()

    for result in model.track(source=args.source,
                              stream=args.stream,
                              agnostic_nms=True,
                              conf=args.conf,
                              verbose=args.verbose):
        frame = result.orig_img
        if frame is None:
            break
        annotator = Annotator(frame, font_size=2, line_width=2)

        for bed in beds:
            annotator.box_label(bed, 'BED', color=(
                86, 168, 227), txt_color=(255, 255, 255))

        for i, r in enumerate(result):
            boxes = r.boxes
            keypoints = r.keypoints.xy[0]
            for box in boxes:
                # get box coordinates in (top, left, bottom, right) format
                human_box = box.xyxy[0]
                c = box.cls
                status = human_status(human_box, keypoints, beds)

                tag_name = f"#{str(i)} {model.names[int(c)]} {HUMAN_STATUS[status]['Description']} {str(r.boxes[0].conf.data[0].item())[0:4]}"
                box_color = HUMAN_STATUS[status]['Color']
                annotator.box_label(human_box, tag_name,
                                    color=box_color, txt_color=(255, 255, 255))

            for keypoint_indx, keypoint in enumerate(keypoints):
                cv2.putText(frame, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame = annotator.result()
        video_result.write(frame)
        if args.show:
            cv2.imshow("yolov8", frame)
            if (cv2.waitKey(30) == 27):
                break
    video_result.release()
    cv2.destroyAllWindows()
    print("The video was successfully saved")


if __name__ == "__main__":
    main()
