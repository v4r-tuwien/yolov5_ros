import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import rospy
from std_msgs.msg import String, Float32MultiArray, Int64
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge, CvBridgeError

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
#from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.get_ros_image import LoadROSImage
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

import numpy as np

import rospy
from std_msgs.msg import Header
from object_detector_msgs.msg import BoundingBox, Detection, Detections
from object_detector_msgs.srv import detectron2_service_server

if __name__ == "__main__":

    device = ''
    device = select_device(device)
    weights = ROOT / 'runs/train-seg/exp14/weights/last.pt' 
    data = ROOT / 'data/ycbv.yaml'

    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)

    bs = 1  # batch_size
    imgsz=(480, 640)
    stride=32
    conf_thres = 0.6
    iou_thres = 0.9
    half = False
    bridge = CvBridge()
    imgsize = check_img_size(imgsz, s=model.stride)

    model.warmup(imgsz=(1 if model.pt else bs, 3, *imgsz))  # warmup

    def detect(req):
        print("request detection...")

        # === IN ===
        # --- rgb
        rgb = req.image
        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480

        try:
            img0 = bridge.imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            print(e)

        img = letterbox(img0, imgsz, stride=stride, auto=model.pt)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        # === DETECTION ===
        im = torch.from_numpy(im).to(device)
        im = im.s() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference    
        pred, proto = model(im, augment=False)[:2]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000, nm=32)

        annotator = Annotator(img0, line_width=3, example=str(model.names))

        # Process predictions
        #obj_ids, rois, masks, scores = maskrcnn.detect(rgb)

        # === OUT ===
        detections = []

        for i, det in enumerate(pred):  # per image
            if len(det):
                detection = Detection()
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()  # rescale boxes to im0 size
                masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], img0.shape[:2])  # HWC
            
                bbox = det[:, :4].cpu().detach().numpy()
                confidence = det[:, :5].cpu().detach().numpy()
                class_label = det[:, :6].cpu().detach().numpy()
                masks = masks.cpu().detach().numpy()

                print(bbox[0])
                print(confidence[0][4])
                print(model.names[int(class_label[0][5])])
                print(class_label[0][5])
                print(masks)
                print(masks[0].shape)

                # ---
                detection.name = model.names[int(class_label[0][5])]
                #detection.name = str(int(class_label[0][5])+1)
                # ---

                # ---            
                bbox_msg = BoundingBox()
                bbox_msg.xmin = int(bbox[0][0])
                bbox_msg.ymin = int(bbox[0][1])
                bbox_msg.xmax = int(bbox[0][2])
                bbox_msg.ymax = int(bbox[0][3])
                detection.bbox = bbox_msg
                # ---
                #TODO mask!
                            # ---
                #mask = masks[0][:, :, i]
                mask = masks[0]
                mask_ids = np.argwhere(mask.reshape((height * width)) > 0)
                detection.mask = list(mask_ids.flat)
                # ---

                # ---
                detection.score = confidence[0][4]
                # ---
                # 

                detections.append(detection)      

        ros_detections = Detections()
        ros_detections.width, ros_detections.height = 640, 480
        ros_detections.detections = detections

        return ros_detections


    rospy.init_node("detection_yolov5")
    s = rospy.Service("detect_objects", detectron2_service_server, detect)
    print("Detection with YOLOv5 ready.")

    rospy.spin()
