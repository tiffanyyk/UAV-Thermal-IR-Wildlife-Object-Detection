# ROB498

## Setup
* [ ] Dockerfile and build script
* [ ] Docker run script
* [ ] Transfer code from their colab notebook into organized folders in this repo
* [ ] Form a team on the challenge website

## Modifications
* [ ] YOLOR Architecture
* [ ] CenterNet

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
    │   MOT to COCO format code here
    └───birdsai
└───src (will likely end up with subfolders)
    └───yolor
        |   all the folders from the yolor repo copied into here
    └───centernet
        |   all the folders from the centernet repo copied into here
└───tools (TBD)
    |   experiment script (i.e. script that starts the training / testing run)
    |   config files (configs for the train / test run)
```
