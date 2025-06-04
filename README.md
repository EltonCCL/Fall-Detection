# AI + Healthcare: Research and Development of Intelligent Systems for Medical Diagnosis and Applications


## Project Overview

This project, undertaken as part of an Undergraduate Research Opportunity Program (UROP), addresses the critical need for reliable fall detection systems, particularly for ensuring the safety and well-being of elderly individuals. The work involves research and development of an intelligent system that utilizes skeleton-based action recognition models for medical diagnosis applications, specifically fall detection. The pipeline for this fall detection model is developed using the Python framework provided by OpenMMLab.

## Methodology

The fall detection process is implemented as a two-part pipeline: pose estimation and action recognition.

### Pose Estimation

A top-down approach is utilized for pose estimation. Videos are first processed by a Faster-RCNN model for human detection. Following detection, human keypoints are extracted from the identified bounding boxes using the HRNet-W32 model.

### Action Recognition

For every four frames in the video, a 16-frame clip is extracted to serve as a sample. Each of these sample clips, containing the extracted pose information, is then passed to one of the selected skeleton-based action recognition models. These models predict the probability of a "fall" occurring.

## Models Evaluated

This study conducted an evaluation of four skeleton-based action recognition models:

ST-GCN (Spatial Temporal Graph Convolutional Networks)

2s-AGCN (Two-Stream Adaptive Graph Convolutional Networks)

PoseC3D

STGCN++

These models were trained using the "falling down" action class (A43) from the NTU RGB-D 60 dataset.

## Datasets Used for Evaluation

The performance of the models was benchmarked using two distinct fall detection datasets:

UR Fall Detection Dataset: This dataset includes 30 sequences of falls and 40 sequences of daily living activities. For this study, only the RGB data from the sequences was utilized for model testing.

Le2i Fall Detection Dataset: This dataset captures realistic video sequences in different environments. Specifically, 70 videos in the coffee room environment and 60 videos in the home environment were used for testing the model.

## Framework and Libraries

The project leverages several key tools and libraries:

OpenMMLab: A comprehensive Python framework that standardizes pipelines for various computer vision tasks, including training, testing, and inferencing. It provides influential computer vision models and standardizes different pipelines.

MMPose: An OpenMMLab toolbox used for building the pose estimation pipeline.

MMAction2: An OpenMMLab toolbox employed for constructing the skeleton-based action recognition pipeline.

Python: The primary programming language.

## Installation

Install OpenMMLab and its components. The installation can be approached in two ways:
1. From Python package: Allows for quick runs but limits customization of the data flow.
2. From source: Recommended for developers who wish to customize features. This method may present challenges with Python dependencies and inter-toolbox configuration.

Please refer to the official OpenMMLab, MMPose, and MMAction2 documentation for detailed installation instructions and troubleshooting, as dependencies can be complex.

## Usage

This project involves a pipeline for fall detection using video input:

1.  Data Preparation: Ensure your video data is accessible and in a format compatible with the pipeline.
2.  Pose Estimation: Process input videos using the MMPose-based pipeline (Faster-RCNN for human detection, HRNet-W32 for keypoint extraction) to generate skeleton data.
3.  Action Recognition: Feed the generated skeleton data (as 16-frame clips sampled every 4 frames) into one of the evaluated action recognition models (ST-GCN, 2s-AGCN, PoseC3D, STGCN++) using MMAction2.
4.  Output: The model will output a "fall" probability for the input clips.

(Detailed commands or script execution steps should be added here if you have specific scripts to run the pipeline.)

## Key Results Summary

The models were evaluated based on Accuracy, Precision, Recall, and F1 Score.

### Evaluation on UR Fall Detection Dataset

ST-GCN model demonstrated the poorest performance, achieving an F1 score of 0.622.

The other three models, 2s-AGCN, PoseC3D, and STGCN++, exhibited identical F1 scores of 0.748.

### Evaluation on Le2i Fall Detection Dataset

ST-GCN did not rank as the lowest-performing model on this dataset and performed marginally better than 2s-AGCN. (Note: Original report Table 2 shows ST-GCN F1 0.844, 2s-AGCN F1 0.811)

PoseC3D achieved an F1 score of 0.860.

STGCN++ achieved an F1 score of 0.851.

### Overall Model Comparison

PoseC3D generally outperformed the other models in terms of the F1 score.

Considering hardware support and computational cost, STGCN++ may be a preferable choice for real-time inference. STGCN++ has 1.39 million parameters and requires 1.95 GFLOPs, while PoseC3D has 2.0 million parameters and requires 20.6 GFLOPs. STGCN++ demands significantly less computation despite a slight loss in performance.

### Model Training Statistics

(From MMAction2, based on NTU RGB-D 60 pre-training)

| Model   | Top 1 acc. | FLOPs | Params |
| :------ | :--------- | :---- | :----- |
| ST-GCN  | 88.95      | 3.8G  | 3.1M   |
| 2s-AGCN | 88.60      | 4.4G  | 3.5M   |
| PoseC3D | 93.6       | 20.6G | 2.0M   |
| STGCN++ | 89.29      | 1.95G | 1.39M  |

## Limitations and Future Work

The study identified certain limitations and proposed areas for future enhancements:

A performance bottleneck was observed, potentially due to the limitations of the top-down approach used for human detection (Faster-RCNN). In scenarios such as videos captured from a top view or during the act of falling, the human recognition model may fail to detect individuals, thereby preventing the subsequent parts of the pipeline from functioning correctly.

Recommendation: It is recommended to re-evaluate the four skeleton-based action recognition models in conjunction with different human pose estimation models, such as those from the YOLO series or the recently proposed RTMPose.

The skeleton-based action recognition models appeared to be less sensitive to falling poses when captured from a top-view angle. All false negative cases in the UR Fall Detection Dataset were from horizontal top-view videos, which might be a consequence of the training data characteristics in the NTU-60 dataset.

Recommendation: To improve performance in such scenarios, fine-tuning the models with additional data featuring falling poses captured from diverse angles is advisable.
