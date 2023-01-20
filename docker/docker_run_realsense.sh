#!/bin/bash

#docker run \
#-it \
#--network=host \
#ros:noetic-ros-base

echo $HOSTNAME

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" --privileged \
--volume="/dev/bus/usb:/dev/bus/usb" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/home/hoenig/iChores/yolov5:/root/yolo" \
--net host \
--env ROS_HOSTNAME=$HOSTNAME \
--env ROS_MASTER_URI=http://$HOSTNAME:11311/ \
--name=realsense_ros yolo_ros

#xhost +local:`docker inspect --format='{{ .Config.Hostname }}' yolo_rosv0`
