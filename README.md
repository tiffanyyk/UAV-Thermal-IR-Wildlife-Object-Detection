# ROB498
## TODOs
* [ ] Setup VSCode with Google Colab - how / what link to follow? (Helen, Aditi)
* [x] Docker scripts (Tiffany)
* [x] Dataset conversion from BIRDSAI to COCO (Isobel, Tiffany)
  * [x] What is the original width, height, number of channels of the thermal images? (698, 520), (698, 519), (640, 512) **we'll need resizing code**
  * [x] Use a subset of BIRDSAI and run `convert_mot_to_coco.py` on it to verify the script works (see [`data/dataset/coco_format`](data/dataset/coco_format) for output files
  * [x] Figure out a common location for all of us to store the dataset, relative to the repo ([`data/dataset`](data/dataset))
* [ ] YOLOR Architecture (Helen, Aditi)
  * Use a subset of COCO to verify the training code works
  * Where is the code for the YOLOR model architecture? Which component needs to be modified for thermal images?
  * How do we run training / pass in configs & hyperparameters?
* [ ] CenterNet2
## Repo Structure 
Rough repository structure we can follow as we build our project
```
ROB498
│   README.md
|   .gitignore
|   LICENSE
└───setup
    └───docker
        |   Dockerfile, build.sh, run.sh, requirements.txt
└───data
    │   convert_birdsai_to_coco.py
    └───utils
    └───dataset
        └───TrainReal (put extracted dataset files here)
        └───TestReal (put extracted dataset files here)
        └───coco_format (output from `convert_mot_to_coco.py`)
            |   TrainReal.json
            |   TestReal.json
└───src (will likely end up with subfolders)
    └───yolor
        |   all the folders from the yolor repo copied into here
    └───centernet2
        |   all the folders from the centernet repo copied into here
└───tools (TBD)
    |   experiment script (i.e. script that starts the training / testing run)
    |   config files (configs for the train / test run)
```
## Running YOLOR
```
# TODO: Merge yolor docker with repo docker

docker run -ti -v /PATH_TO_DATASET/data/dataset/birsai:/birdsai -v /PATH_TO_REPO/ROB498/src/yolor:/yolor --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3

python train.py --batch-size 16 --img 640 640 --data birdsai_helen.yaml --cfg cfg/yolor_p6_birdsai.cfg --weights 'yolor_p6.pt' --device 0 --name yolor_p6 --hyp hyp.scratch.640.yaml --epochs 100
```

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

