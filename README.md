# YOLOv5 for CaringRobots
YOLOv5 with ROS implementation for workshop task of CaringRobotis project  
fleckerl.yaml dataset with the following objects:  

0: coil
1: glue
2: marker
3: measuring_tape
4: multimeter
5: paint_brush
6: pliers
7: ruler
8: screwdriver
9: tape
10: wood_glue  

## Dataset generation:
- 20-50 images taken from each object from various angles
- generate masks with [`Dino-Vit-Features`](https://github.com/ShirAmir/dino-vit-features) algorithm for class agnostic object segmentation  
- use masks and images to generate dataset via [`pasting random object crops onto random backgrounds`](https://medium.com/@alexppppp/how-to-create-synthetic-dataset-for-computer-vision-object-detection-fd8ab2fa5249) 
- a dataset of 50 000 images for training and 5000 images for validation is generated

## Training:
(Trained with RTX 3090 24GB)  

`python train.py --epochs 300 --data data/fleckerl.yaml --batch-size 128`

## Inference:
Start realsense2 camera node:  
`roslaunch realsense2_camera rs_camera.launch`
Start python YOLOv5 rosnode:  
`python detect_ros.py --weights runs/train/exp39/weights/last.pt --data data/fleckerl.yaml --conf-thres 0.7 --camera-topic /camera/color/image_raw`

## Docker:
copy the runs folder into the root directory of the the cloned yolo directory  

`bash docker_build.sh`  
`bash docker_run.sh`  -> cd yolo; python detect_ros.py --weights runs/train/exp39/weights/last.pt --data data/fleckerl.yaml --conf-thres 0.7 --camera-topic /camera/color/image_raw
`bash docker_run_realsense.sh`  -> roslaunch realsense2_camera rs_camera.launch  

Open another window and execute: `` xhost +local:`docker inspect --format='{{ .Config.Hostname }}' yolo_rosv0` ``






