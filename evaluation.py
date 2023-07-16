import json
from pathlib import Path

UR_FALL_DIR = "dataset/UR Fall Detection Dataset"
MULTICAM_DIR = "dataset/Multicam"
LE2I_DIR = "dataset/Le2i Fall Detection Dataset"
FALL_DETECTION_DIR = "dataset/Fall Detection Dataset"

MULTI_CAM_GROUND = "/Users/eltonli/Desktop/dataset/Multicam/ground_truth.json"


class ModelResult:
    def __init__(self) -> None:
        self.total_n = 0
        self.true_pos = []
        self.true_neg = []
        self.false_pos = []
        self.false_neg = []
        pass

    def add_tp(self, vid_name):
        self.total_n += 1
        self.true_pos.append(vid_name)
        pass

    def add_tn(self, vid_name):
        self.total_n += 1
        self.true_neg.append(vid_name)
        pass

    def add_fp(self, vid_name):
        self.total_n += 1
        self.false_pos.append(vid_name)
        pass

    def add_fn(self, vid_name):
        self.total_n += 1
        self.false_neg.append(vid_name)
        pass

    def get_dict(self):
        precision = len(self.true_pos) / (len(self.true_pos) + len(self.false_pos))
        recall = len(self.true_pos) / (len(self.true_pos) + len(self.false_neg))
        f1_score = 2 * precision * recall / (precision + recall)

        return_result = {
            "score": {
                "total": self.total_n,
                "true_pos": len(self.true_pos),
                "true_neg": len(self.true_neg),
                "false_pos": len(self.false_pos),
                "false_neg": len(self.false_neg),
                "accuracy": len(self.true_pos) / (self.total_n),
                "precison": precision,
                "recall": recall,
                "f1_score": f1_score,
            },
            "file": {
                "true_pos": self.true_pos,
                "true_neg": self.true_neg,
                "false_pos": self.false_pos,
                "false_neg": self.false_neg,
            },
        }

        return return_result


def video_fall():
    pass


def le2i():
    pass


def multicam():
    file = open(MULTI_CAM_GROUND)
    data = json.load(file)
    file.close()
    ground_truth = {}
    for vid in data:
        temp = [i for i in range(data[vid][0]["start"] + 1)]
        for interval in data[vid]:
            temp += [i for i in range(interval["start"], interval["end"] + 1)]
        print(temp)
        break
    pass


def ur_fall():
    result = ModelResult()

    fall_video_dir = UR_FALL_DIR / Path("fall")
    for fall_video in fall_video_dir.glob("*"):
        # Call fall detection
        det_result = [0, 0]
        if 1 in det_result:
            result.add_tp(fall_video)
        else:
            result.add_fn(fall_video)

    daily_video_dir = UR_FALL_DIR / Path("adl")
    for fall_video in daily_video_dir.glob("*"):
        # Call fall detection
        det_result = [0, 0]
        if 1 in det_result:
            result.add_fp(fall_video)
        else:
            result.add_tn(fall_video)

    stat_data = result.get_dict()
    print(stat_data['score'])


def main():
    ur_fall()


if __name__ == "__main__":
    main()
