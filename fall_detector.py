# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import tempfile

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction
from mmengine.utils import track_iter_progress

from mmaction.apis import (
    detection_inference,
    inference_recognizer,
    init_recognizer,
    pose_inference,
)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError("Please install moviepy to enable output file")

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.4
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description="MMAction2 demo")
    parser.add_argument(
        "--video",
        help="video file/url",
        default="testing_video/Untitled video - Made with Clipchamp.mp4",
    )
    parser.add_argument(
        "--out_filename", help="output filename", default="demo/default_out.mp4"
    )
    parser.add_argument(
        "--config",
        # default=(
        #     "configs/skeleton/posec3d/"
        #     "slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py"
        # ),
        default=(
            "configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
        ),
        help="skeleton model config file path",
    )
    parser.add_argument(
        "--checkpoint",
        # default=(
        #     "https://download.openmmlab.com/mmaction/skeleton/posec3d/"
        #     "slowonly_r50_u48_240e_ntu60_xsub_keypoint/"
        #     "slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth"
        # ),
        default=(
            "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth"
        ),
        help="skeleton model checkpoint file/url",
    )
    parser.add_argument(
        "--det-config",
        default="demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py",
        help="human detection config file path (from mmdet)",
    )
    parser.add_argument(
        "--det-checkpoint",
        default=(
            "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
            "faster_rcnn_r50_fpn_2x_coco/"
            "faster_rcnn_r50_fpn_2x_coco_"
            "bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
        ),
        help="human detection checkpoint file/url",
    )
    parser.add_argument(
        "--det-score-thr",
        type=float,
        default=0.9,
        help="the threshold of human detection score",
    )
    parser.add_argument(
        "--det-cat-id", type=int, default=0, help="the category id for human detection"
    )
    parser.add_argument(
        "--pose-config",
        default="demo/demo_configs/" "td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py",
        help="human pose estimation config file path (from mmpose)",
    )
    parser.add_argument(
        "--pose-checkpoint",
        default=(
            "https://download.openmmlab.com/mmpose/top_down/hrnet/"
            "hrnet_w32_coco_256x192-c78dce93_20200708.pth"
        ),
        help="human pose estimation checkpoint file/url",
    )
    parser.add_argument(
        "--label-map",
        default="tools/data/skeleton/label_map_ntu60.txt",
        help="label map file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="CPU/CUDA device option"
    )
    parser.add_argument(
        "--short-side",
        type=int,
        default=480,
        help="specify the short-side length of the image",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. For example, "
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'",
    )
    parser.add_argument(
        "--predict-stepsize",
        default=4,
        type=int,
        help="give out a prediction per n frames",
    )
    parser.add_argument(
        "--output-stepsize",
        default=2,
        type=int,
        help=(
            "show one frame per n frames in the demo, we should have: "
            "predict_stepsize % output_stepsize == 0"
        ),
    )
    parser.add_argument(
        "--output-fps", default=6, type=int, help="the fps of demo video output"
    )
    parser.add_argument(
        "--action-score-thr",
        type=float,
        default=0.1,
        help="the threshold of human action score",
    )
    args = parser.parse_args()
    return args


SAMPLER = {
    "type": "SampleAVAFrames",
    "clip_len": 16,
    "frame_interval": 1,
    "test_mode": True,
}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find("(") != -1:
        st, ed = name.find("("), name.find(")")
        name = name[:st] + "..." + name[ed + 1 :]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.
    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            # (prop.data.cpu().numpy(), [x[0] for x in res], [x[1] for x in res])
            (prop, [x[0] for x in res], [x[1] for x in res])
        )
    return results


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = "03045e-023e8a-0077b6-0096c7-00b4d8-48cae4"
plate_blue = plate_blue.split("-")
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = "004b23-006400-007200-008000-38b000-70e000"
plate_green = plate_green.split("-")
plate_green = [hex2color(h) for h in plate_green]


def visualize(frames, annotations, plate=plate_green, max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_out = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_out[ind]
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                if score[0] >= 0.3:
                    cv2.rectangle(frame, st, ed, (102, 0, 255), 2)
                else:
                    cv2.rectangle(frame, st, ed, plate[0], 2)
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ": ".join([text, f"{score[k]:>.2f}"])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE, THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(
                        frame,
                        text,
                        location,
                        FONTFACE,
                        FONTSCALE,
                        FONTCOLOR,
                        THICKNESS,
                        LINETYPE,
                    )

    return frames_out


def split_person(ipt_list):
    persons = []
    for i, n in enumerate(ipt_list):
        frame = []
        if len(n) == 0:
            frame.append(n)
        else:
            for p in n:
                frame.append(p)
        persons.append(frame)
    return persons


def main():
    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, frames = frame_extract(args.video, args.short_side, tmp_dir.name)

    num_frame = len(frame_paths)
    h, w, _ = frames[0].shape

    # Get Human detection results.
    det_results, _ = detection_inference(
        args.det_config,
        args.det_checkpoint,
        frame_paths,
        args.det_score_thr,
        args.det_cat_id,
        args.device,
    )
    torch.cuda.empty_cache()
    # det_results = [np.delete(x, np.s_[1:], axis=0) for x in det_results]
    # Get Pose estimation results.
    # split_det_results = split_person(det_results)
    # for person in split_det_results:
    pose_results, pose_data_samples = pose_inference(
        args.pose_config, args.pose_checkpoint, frame_paths, det_results, args.device
    )
    torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir="",
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality="Pose",
        total_frames=num_frame,
    )
    num_person = max([len(x["keypoints"]) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_frame, num_person, num_keypoint, 2), dtype=np.float16)
    keypoint_score = np.zeros((num_frame, num_person, num_keypoint), dtype=np.float16)
    for i, poses in enumerate(pose_results):
        poses["keypoints"] = np.append(
            poses["keypoints"],
            np.zeros(
                (num_person - len(poses["keypoints"]), num_keypoint, 2),
                dtype=np.float32,
            ),
            axis=0,
        )
        poses["keypoint_scores"] = np.append(
            poses["keypoint_scores"],
            np.zeros(
                (num_person - len(poses["keypoint_scores"]), num_keypoint),
                dtype=np.float32,
            ),
            axis=0,
        )

        keypoint[i] = poses["keypoints"]
        keypoint_score[i] = poses["keypoint_scores"]
    # keypoint = np.delete(keypoint, np.s_[1:], axis=1)
    # keypoint_score = np.delete(keypoint_score, np.s_[1:], axis=1)
    # fake_anno["keypoint"] = keypoint.transpose((1, 0, 2, 3))
    # fake_anno["keypoint_score"] = keypoint_score.transpose((1, 0, 2))

    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)

    model = init_recognizer(config, args.checkpoint, args.device)

    # -----------------
    # SAMPLER = {
    #     "type": "SampleAVAFrames",
    #     "clip_len": 8,
    #     "frame_interval": 8,
    #     "test_mode": True,
    # }

    clip_len, frame_interval = SAMPLER["clip_len"], SAMPLER["frame_interval"]
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, "We would like to have an even clip_len"
    # Note that it's 1 based here
    timestamps = np.arange(
        window_size // 2, num_frame + 1 - window_size // 2, args.predict_stepsize
    )
    test = [i for i in range(1, num_frame)]
    predictions = []
    print("Performing SpatioTemporal Action Detection for each clip")
    prog_bar = mmengine.ProgressBar(len(timestamps))
    for timestamp, proposal in zip(timestamps, det_results):
        # if proposal.shape[0] == 0:
        #     predictions.append(None)
        #     prog_bar.update()
        #     continue
        # print(timestamp)
        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        # print(frame_inds)

        # for each person
        # Change fake_anno

        kp = keypoint[frame_inds].transpose((1, 0, 2, 3))
        kps = keypoint_score[frame_inds].transpose((1, 0, 2))

        with torch.no_grad():
            prediction = []
            if len(kp) == 0:
                prediction.append([])
            else:
                for i in range(len(kp)):
                    prediction.append([])
                    fake_anno = dict(
                        frame_dir="",
                        label=-1,
                        img_shape=(h, w),
                        original_shape=(h, w),
                        start_index=0,
                        modality="Pose",
                        total_frames=len(frame_inds),
                        keypoint=kp[i : i + 1],
                        keypoint_score=kps[i : i + 1],
                    )

                    # fake_anno['total_frames'] = len(frame_inds)
                    # fake_anno["keypoint"] = keypoint[frame_inds].transpose((1, 0, 2, 3))
                    # fake_anno["keypoint_score"] = keypoint_score[frame_inds].transpose((1, 0, 2))

                    result = inference_recognizer(model, fake_anno)
                    label_map = [x.strip() for x in open(args.label_map).readlines()]
                    prediction[i].append(
                        (label_map[42], result.pred_scores.item[42].item())
                    )

                    # for p, n in enumerate(result.pred_scores.item):
                    #     if n > args.action_score_thr:
                    #         prediction[i].append((label_map[p], n.item()))

            predictions.append(prediction)
        prog_bar.update()
    pass

    results = []
    center_human_detection = [det_results[ind - 1] for ind in timestamps]
    for human_detection, prediction in zip(center_human_detection, predictions):
        results.append(pack_result(human_detection, prediction, h, w))

    def dense_timestamps(timestamps, n):
        """Make it nx frames."""
        old_frame_interval = timestamps[1] - timestamps[0]
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int64)

    dense_n = int(args.predict_stepsize / args.output_stepsize)
    frames = [
        cv2.imread(frame_paths[i - 1]) for i in dense_timestamps(timestamps, dense_n)
    ]

    print("Performing visualization")
    vis_frames = visualize(frames, results)
    vid = mpy.ImageSequenceClip(
        [x[:, :, ::-1] for x in vis_frames], fps=args.output_fps
    )
    vid.write_videofile(args.out_filename)
    # -----------------
    # result = inference_recognizer(model, fake_anno)

    # max_pred_index = result.pred_scores.item.argmax().item()
    # label_map = [x.strip() for x in open(args.label_map).readlines()]
    # action_label = label_map[max_pred_index]

    # visualize(args, frames, pose_data_samples, action_label)

    tmp_dir.cleanup()


if __name__ == "__main__":
    main()
