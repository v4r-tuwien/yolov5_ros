#!/bin/bash

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" --privileged \
--volume="/dev/bus/usb:/dev/bus/usb" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/home/hoenig/iChores/yolov5:/root/yolo" \
--name=yolo_rosv0 yolo_ros

#xhost +local:`docker inspect --format='{{ .Config.Hostname }}' yolo_rosv0`
