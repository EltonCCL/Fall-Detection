import json
from pathlib import Path
import os
from fall_detector import inference
from tqdm import tqdm

UR_FALL_DIR = "dataset/UR Fall Detection Dataset"
MULTICAM_DIR = "dataset/Multicam"
LE2I_DIR = "dataset/Le2i Fall Detection Dataset"
FALL_DETECTION_DIR = "dataset/Fall Detection Dataset"

MULTI_CAM_GROUND = "/Users/eltonli/Desktop/dataset/Multicam/ground_truth.json"


def le2i():
    sub_dir = ["coffee_room", "home"]
    model_output = {}
    for sub in sub_dir:
        model_output[sub] = []

        daily_video_dir = LE2I_DIR / Path(f"{sub}/Videos")
        for fall_video in daily_video_dir.glob("*"):
            output_vid = str(fall_video).replace('dataset', 'output')
            dn = os.path.abspath(output_vid)
            output_dir = os.path.dirname(dn)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            print(output_vid)
            print(output_vid)
            print(output_dir)
            print()
            model_output[sub].append(
                {str(fall_video).split("/")[-1]: inference(str(fall_video), str(output_vid), device='cpu')})
        print(len(model_output[sub]))

    json_object = json.dumps(model_output, indent=4)
    with open("inference_result/le2i.json", "w") as outfile:
        outfile.write(json_object)


def multicam():
    model_output = {}

    chutes = Path(MULTICAM_DIR)
    for chute in chutes.glob("chute*"):
        chut_idx = str(chute).split("/")[-1]
        model_output[chut_idx] = {}
        for cam in chute.glob("*"):
            cam_idx = str(cam).split("/")[-1]
            output_vid = str(cam).replace('dataset', 'output')
            dn = os.path.abspath(output_vid)
            output_dir = os.path.dirname(dn)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            print(cam)
            print(output_vid)
            print(output_dir)
            print()
            model_output[chut_idx][cam_idx] = inference(str(cam), str(output_vid), device='cpu')

    json_object = json.dumps(model_output, indent=4)
    with open("inference_result/multi_cam.json", "w") as outfile:
        outfile.write(json_object)


def ur_fall():
    sub_dir = ["fall", "adl"]
    model_output = {}
    for sub in sub_dir:
        model_output[sub] = []

        daily_video_dir = UR_FALL_DIR / Path(f"{sub}")
        for fall_video in daily_video_dir.glob("*"):

            output_vid = str(fall_video).replace('dataset', 'output')
            dn = os.path.abspath(output_vid)
            output_dir = os.path.dirname(dn)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            model_output[sub].append(
                {str(fall_video).split("/")[-1]: inference(str(fall_video), str(output_vid), device='cpu')})
        # print(len(model_output[sub]))

    json_object = json.dumps(model_output, indent=4)
    with open("inference_result/UR_FALL.json", "w") as outfile:
        outfile.write(json_object)


def main():
    le2i()


if __name__ == "__main__":
    main()
