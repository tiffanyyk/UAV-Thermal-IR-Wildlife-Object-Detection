# ROB498 - Wildlife Protection Through Aerial Drone Surveillance
The goal of our project was to build a detection system that can recognize animals in thermal infrared aerial images. We investigated YOLOR and YOLOv5, and performed several experiments on the BIRDSAI dataset before arriving at our final design, shown in the figure below. The final prototype exceeded our design requirement with a ($mAP$) of 38.2\% and is well within the hardware constraints of the GPU assumed to be available on the UAV. More details can be found in our [report](assets/ROB498_Final_Report.pdf).
<p align="center"><img src="assets/design_diagram.png"/></p>

## Repository Structure 
<details>
<summary>[Click to view]</summary>

```
ROB498/
│   README.md
|   .gitignore
|   LICENSE
└───setup/
    └───docker/
        |   Dockerfile, build_docker.sh, run_docker_gpu.sh, run_docker.sh, py_interpreter.sh
└───data/
    │   convert_birdsai_to_coco.py -- script to convert BIRDSAI dataset to COCO format
    |   count_classes.py -- script to determine the number of samples in each object class
    |   Anchor_Statistics.ipynb -- K-means to determine anchors for YOLOR
    └───utils/ -- folder containing code for BIRDSAI data handling which we didn't use
    └───dataset/ (store dataset files here)
        └───TrainReal/
            └───annotations/
            └───images/
        └───TestReal/
            └───annotations/
            └───images/
        └───TrainSimulation/
            └───annotations/
            └───images/
        └───coco_format/ (data/convert_birdsai_to_coco.py will save outputs to this directory)
            |   TrainReal.json
            |   TestReal.json
            |   TrainSimulation.json
└───src/
    └───common/
        |   data_utils.py -- data-related utility functions shared between models
        |   general_utils.py -- general utility functions shared between models
    └───yolor/
        |   YOLOR specific code lives in this directory
    └───yolov5/
        |   YOLOv5 specific code lives in this directory
```
</details>

## Environment Setup
<details>
<summary>[Click to view]</summary>
To replicate the results in our report, we recommend using the Docker setup provided in this repository. All the relevant files are located in [`setup/docker`](setup/docker). To set this up, do the following:
```
$ cd setup/docker
```

### 1. Build the Docker image
Running the following command will build a docker image with the image name and tag specified in lines 4-5 of [`build_docker.sh`](setup/docker/build_docker.sh#l4). Modify this according to your preference. The default is currently set to `tiffanyyk/tiffanyyk:rob498-yolo`.
```
$ ./build_docker.sh
```
### 2. Start a Docker container
This will start a docker container using the image you have just built. If you changed the name of the docker image in [`build_docker.sh`](setup/docker/build_docker.sh#l4), modify lines 5-7 of [`run_docker_gpu.sh`](setup/docker/run_docker_gpu.sh#l5) and [`run_docker.sh`](setup/docker/run_docker.sh#l5) accordingly.

If you have a gpu on your system:
```
$ ./run_docker_gpu.sh
```
If you do not have a gpu:
```
$ ./run_docker.sh
```
Note that running the code without GPU is not recommended.

### Remote Debugging
If this is relevant to you, the [`py_interpreter.sh`](setup/docker/py_interpreter.sh) script is provided for remote debugging setup.

### Setup Notes
If you encounter any permission errors when building the image or running the docker container, use `chmod +x [build or run script]`.

</details>

## Training and Testing YOLOR
<details>
<summary>[Click to view]</summary>

```
# Training YOLOR with 3 classes on real data only
$ python train.py --batch-size 16 --img 640 640 --data birdsai_3class.yaml --cfg cfg/yolor_p6_birdsai_3class.cfg --weights '' --device 0 --name yolor_p6 --hyp hyp.scratch.640.yaml --epochs 100

# Testing YOLOR with 3 classes on real data only
python test.py --data birdsai_3class.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yolor_p6_birdsai_3class.cfg --weights '/path/to/saved/checkpoint.pt' --name yolor_p6_val --verbose --names data/birdsai_3class.names
```
</details>

## Training and Testing YOLOv5

## Running YOLOR & YOLOv5
```
# TODO: Merge yolor docker with repo docker

docker run -ti -v /PATH_TO_DATASET/data/dataset/birsai:/birdsai -v /PATH_TO_REPO/ROB498/src/yolor:/yolor --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3

python train.py --batch-size 16 --img 640 640 --data birdsai_helen.yaml --cfg cfg/yolor_p6_birdsai.cfg --weights 'yolor_p6.pt' --device 0 --name yolor_p6 --hyp hyp.scratch.640.yaml --epochs 100

# Training YOLOR with 3 classes & real data only
python train.py --batch-size 16 --img 640 640 --data birdsai_3class.yaml --cfg cfg/yolor_p6_birdsai_3class.cfg --weights '' --device 0 --name yolor_p6 --hyp hyp.scratch.640.yaml --epochs 100

# Distributed training
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --batch-size 16 --img 640 640 --data birdsai_3class.yaml --cfg cfg/yolor_p6_birdsai_3class.cfg --weights '' --device 0,1 --sync-bn --name yolor_p6_birdsai --hyp hyp.scratch.640.yaml --epochs 5


# Testing YOLOR with 3 classes
python test.py --data birdsai_3class.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yolor_p6_birdsai_3class.cfg --weights '/h/helen/school/ROB498/runs/train/yolor_train_3class_1/weights/last.pt' --name yolor_p6_val --verbose --names data/birdsai_3class.names

python test.py --data birdsai_2class.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yolor_p6_birdsai_2class_origanchors.cfg --weights '/h/helen/school/ROB498/runs/train/yolor_train_2class_12/weights/best_ap50.pt' --name yolor_2class_origanchors_3channel_val --verbose --names data/birdsai_2class.names

python test.py --data birdsai_3class.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yolor_p6_birdsai_3class.cfg --weights '/h/helen/school/ROB498_old/runs/train/yolor_train_3class_32/weights/best_ap50.pt' --name yolor_3class_origanchors_3channel_val --verbose --names data/birdsai_3class.names

python test.py --data birdsai_10class.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yolor_p6_birdsai.cfg --weights '/h/helen/school/ROB498/runs/train/yolor_train_10class_1/weights/best_ap50.pt' --name yolor_10class_origanchors_3channel_val --verbose --names data/birdsai.names


python test.py --data birdsai_2class.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yolor_p6_birdsai_2class_origanchors_1channel.cfg --weights '/h/helen/school/ROB498/runs/train/yolor_train_2class_12/weights/best_ap50.pt' --name yolor_2class_origanchors_1channel_val --verbose --names data/birdsai_2class.names --channels=1

# For plotting stuff for yolor, run in yolor folder
python test.py --data birdsai_2class.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0 --cfg cfg/yolor_p6_birdsai_2class_origanchors.cfg --weights '/h/helen/school/ROB498/runs/train/yolor_train_2class_12/weights/best_ap50.pt' --name yolor_2class_origanchors_3channel_val --verbose --names data/birdsai_2class.names

# For getting inference speed for yolov5, run in yolov5 folder
python val.py --weights /h/helen/school/ROB498/src/yolov5/runs/train/exp5/weights/best.pt --data birdsai_2class.yaml --img 640 --task speed
```

## Acknowledgements
We would like to thank the authors of [YOLOR](https://github.com/WongKinYiu/yolor) and [YOLOv5](https://github.com/ultralytics/yolov5) for open-sourcing their code.
