# YOLOv5 with Docker Container and ROS Wrapper
YOLOv5 for YCB-V object detection:

1. "002_master_chef_can"
2. "003_cracker_box"
3. "004_sugar_box"
4. "005_tomato_soup_can"
5. "006_mustard_bottle"
6. "007_tuna_fish_can"
7. "008_pudding_box"
8. "009_gelatin_box"
9. "010_potted_meat_can"
10. "011_banana"
11. "019_pitcher_base"
12. "021_bleach_cleanser"
13. "024_bowl"
14. "025_mug"
15. "035_power_drill"
16. "036_wood_block"
17. "037_scissors"
18. "040_large_marker"
19. "051_large_clamp"
20. "052_extra_large_clamp"
21. "061_foam_brick"

## Use YCB-V weights:
* `wget -O yolov5_ycbv_weights.zip "https://owncloud.tuwien.ac.at/index.php/s/lbnwdUAR3uSpKOj/download"`
* `unzip yolov5_ycbv_weights.zip`
* `cp -r yolov5_ycbv_weights.pt ./src/yolov5/yolov5_ycbv_weights.pt`
* `rm -r yolov5_ycbv_weights.zip yolov5_ycbv_weights.pt`

## Training:
`python train.py --epochs 300 --data data/your_dataset.yaml`

## Inference:
Start realsense2 camera node:  
`roslaunch realsense2_camera rs_camera.launch`
Start python YOLOv5 rosnode:  
`python detect_ros.py --weights runs/train/exp41/weights/last.pt --data data/your_dataset.yaml --conf-thres 0.8 --iou-thres 0.6 --camera-topic /camera/color/image_raw`

## Docker:
* `cd docker`
* `docker-compose up`

