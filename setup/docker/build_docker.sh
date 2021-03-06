#!/bin/bash
readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}"c)")
DOCKER_REGISTRY=uav-tir-wildlife-od
IMAGE_NAME=${DOCKER_REGISTRY}/uav-tir-wildlife-od
TAG=rob498-yolo
BASE_IMG=nvcr.io/nvidia/pytorch:20.11-py3
DOCKER_FILENAME=Dockerfile
DOCKER_USER=rob498
DOCKER_PASSWORD=rob498

docker build -t ${IMAGE_NAME}:${TAG} \
    -f ${DOCKER_FILENAME} \
    --network host \
    --build-arg BASE_IMG=${BASE_IMG} \
    --build-arg USERNAME=${DOCKER_USER} \
    --build-arg PASSWORD=${DOCKER_PASSWORD} \
    --build-arg GROUP_ID=$(id -g ${USER}) \
    --build-arg USER_ID=$(id -u ${USER}) \
    ${SCRIPT_DIR}
