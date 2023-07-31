import json
from pathlib import Path
import os
from fall_detector import inference
from tqdm import tqdm

OUTPUTDIR = "inference_result_AGCN"
VID_OUT_DIR = "output_AGCN"
UR_FALL_DIR = "dataset/UR_Fall_Detection_Dataset"
MULTICAM_DIR = "dataset/Multicam"
LE2I_DIR = "dataset/Le2i_Fall_Detection_Dataset"

MULTI_CAM_GROUND = "/Users/eltonli/Desktop/dataset/Multicam/ground_truth.json"

CONFIG = f"mmlab/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
CHECKPOINT = f"https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221222-4c0ed77e.pth"

isExist = os.path.exists(OUTPUTDIR)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(OUTPUTDIR)


def le2i():
    sub_dir = ["coffee_room", "home"]
    model_output = {}
    for sub in sub_dir:
        model_output[sub] = []

        daily_video_dir = LE2I_DIR / Path(f"{sub}/Videos")
        for fall_video in daily_video_dir.glob("*"):
            output_vid = str(fall_video).replace("dataset", VID_OUT_DIR)
            dn = os.path.abspath(output_vid)
            output_dir = os.path.dirname(dn)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            # print(output_vid)
            # print(output_vid)
            # print(output_dir)
            # print()
            print(f"------------- {fall_video} -------------")
            model_output[sub].append(
                {
                    str(fall_video).split("/")[-1]: inference(
                        str(fall_video),
                        str(output_vid),
                        config=CONFIG,
                        checkpoint=CHECKPOINT,
                    )
                }
            )
        print(len(model_output[sub]))
        json_object = json.dumps(model_output, indent=4)
        with open(f"{OUTPUTDIR}/le2i.json", "w") as outfile:
            outfile.write(json_object)


def multicam():
    model_output = {}

    chutes = Path(MULTICAM_DIR)
    for chute in chutes.glob("chute*"):
        chut_idx = str(chute).split("/")[-1]
        model_output[chut_idx] = {}
        for cam in chute.glob("*"):
            cam_idx = str(cam).split("/")[-1]
            output_vid = str(cam).replace("dataset", "output")
            dn = os.path.abspath(output_vid)
            output_dir = os.path.dirname(dn)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            print(cam)
            print(output_vid)
            print(output_dir)
            print()
            print(f"------------- {cam} -------------")
            model_output[chut_idx][cam_idx] = inference(
                str(cam), str(output_vid), config=CONFIG, checkpoint=CHECKPOINT
            )
        json_object = json.dumps(model_output, indent=4)
        with open(f"{OUTPUTDIR}/multi_cam.json", "w") as outfile:
            outfile.write(json_object)


def ur_fall(py, ckpt, vid_out_dir, output_dir):
    sub_dir = ["fall", "adl"]
    model_output = {}
    for sub in sub_dir:
        model_output[sub] = []

        daily_video_dir = UR_FALL_DIR / Path(f"{sub}")
        for fall_video in daily_video_dir.glob("*"):
            output_vid = str(fall_video).replace("dataset", vid_out_dir)
            dn = os.path.abspath(output_vid)
            output_dir = os.path.dirname(dn)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            print(f"------------- {fall_video} -------------")
            model_output[sub].append(
                {
                    str(fall_video).split("/")[-1]: inference(
                        str(fall_video), str(output_vid), config=py, checkpoint=ckpt
                    )
                }
            )
        # print(len(model_output[sub]))

        json_object = json.dumps(model_output, indent=4)
        with open(f"{output_dir}/UR_FALL.json", "w") as outfile:
            outfile.write(json_object)


def main():
    # print('------------- Inferencing Multicam -------------')
    # multicam()

    # print('------------- Inferencing Le2i -------------')
    # le2i()

    # print('------------- Inferencing UR FALL -------------')
    # ur_fall()


    ur_fall(
        "mmlab/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py",
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221222-4c0ed77e.pth",
        "output_AGCN",
        "inference_result_AGCN",
    )
    # ur_fall(
    #     "mmlab/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py",
    #     "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth",
    #     "output_stgcn",
    #     "inference_result_stgcn",
    # )
    # ur_fall(
    #     "mmlab/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py",
    #     "https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth",
    #     "output_pose3d",
    #     "inference_result_pose3d",
    # )
    # ur_fall(
    #     "mmlab/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py",
    #     "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth",
    #     "output_plusplus",
    #     "inference_result_plusplus",
    # )


if __name__ == "__main__":
    main()
