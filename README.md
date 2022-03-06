# ROB498

## Setup
* [ ] Dockerfile and build script
* [ ] Docker run script
* [ ] Transfer code from their colab notebook into organized folders in this repo
* [ ] Form a team on the challenge website

## TODOs
* [ ] Setup VSCode with Google Colab - how / what link to follow? (Helen, Aditi)
* [ ] Docker scripts (Tiffany)
* [ ] Dataset conversion from BIRDSAI to COCO (Isobel, Tiffany)
  * What is the original width, height, number of channels of the thermal images?
  * Use a subset of BIRDSAI and run `convert_mot_to_coco.py` on it to verify the script works
  * Figure out a common location for all of us to store the dataset, relative to the repo (see hardcoded paths in conversion script)
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
    │   MOT to COCO format code here
    └───birdsai
└───src (will likely end up with subfolders)
    └───yolor
        |   all the folders from the yolor repo copied into here
    └───centernet2
        |   all the folders from the centernet repo copied into here
└───tools (TBD)
    |   experiment script (i.e. script that starts the training / testing run)
    |   config files (configs for the train / test run)
```
