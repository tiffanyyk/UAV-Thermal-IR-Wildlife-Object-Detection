#!/bin/bash
set -e

# docker names
DOCKER_REGISTRY=uav-tir-wildlife-od
IMAGE_NAME=${DOCKER_REGISTRY}/uav-tir-wildlife-od
TAG=rob498-yolo
CONTAINER_NAME=default_name
WORKING_DIR=/data/workspace #${HOME}

## mount paths
# local paths (in ubuntu) - location where this repo is cloned
WORKSPACE_LOCAL=/mnt/d/year_4/rob498

# paths in container
WORKSPACE_CONTAINER=/data/workspace
DOCKER_CODE_FOLDER=${WORKSPACE_CONTAINER}/ROB498

# resources
MEMORY_LIMIT=30g
NUM_CPUS=6
INTERACTIVE=1
REMOTE=0
VM_PORT=6000

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key=$1
  case $key in
  -i | --interactive)
    INTERACTIVE=$2
    shift
    shift
    ;;
  -wd | --working_directory)
    WORKING_DIR=$2
    shift
    shift
    ;;
  -m | --memory_limit)
    MEMORY_LIMIT=$2
    shift
    shift
    ;;
  -cn | --container_name)
    CONTAINER_NAME=$2
    shift
    shift
    ;;
  -im | --image)
    IMAGE_NAME=${DOCKER_REGISTRY}/$2
    shift
    shift
    ;;
  -t | --tag)
    TAG=$2
    shift
    shift
    ;;
  -nc | --cpus)
    NUM_CPUS=$2
    shift
    shift
    ;;
  -gd | --gpu_device)
    GPU_DEVICE=$2
    shift
    shift
    ;;
  -r | --remote)
    REMOTE=$2
    shift
    shift
    ;;
  -vp | --map_vm_port)
    VM_PORT=$2
    shift
    shift
    ;;
  esac
done

if [[ INTERACTIVE -eq 1 ]]; then
  echo "Running docker in interactive mode"
  IT=-it
else
  IT=-it  # hard code as interactive for now
fi


if [[ REMOTE -eq 1 ]]; then
  IT=-itd
  REMOTE_PORT_MAP="${VM_PORT}:22"
  NETWORK="bridge"
  # need to run container as root for ssh
  # NV_GPU=${GPU_DEVICE} docker run --gpus '"device='"${GPU_DEVICE}"'"' --rm ${IT:-} \
  docker run --rm ${IT:-} \
    --mount type=bind,source=${WORKSPACE_LOCAL},target=${WORKSPACE_CONTAINER} \
    -m ${MEMORY_LIMIT} \
    -w ${WORKING_DIR} \
    --name ${CONTAINER_NAME} \
    --cpus ${NUM_CPUS} \
    --shm-size=6g \
    -p ${REMOTE_PORT_MAP:-"0:0"} \
    --net=${NETWORK:-host} \
    ${IMAGE_NAME}:${TAG}
else
  # NV_GPU=${GPU_DEVICE} docker run --gpus '"device='"${GPU_DEVICE}"'"' --rm ${IT:-} \
  docker run --rm ${IT:-} \
    --mount type=bind,source=${WORKSPACE_LOCAL},target=${WORKSPACE_CONTAINER} \
    -m ${MEMORY_LIMIT} \
    -w ${WORKING_DIR} \
    -e USER=${USER} \
    -e CODE_FOLDER=${DOCKER_CODE_FOLDER} \
    -u $(id -u):$(id -g) \
    --name ${CONTAINER_NAME} \
    --cpus ${NUM_CPUS} \
    --shm-size=6g \
    --net=${NETWORK:-host} \
    ${IMAGE_NAME}:${TAG}
fi
