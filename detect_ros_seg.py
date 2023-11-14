# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

#python detect_ros_seg.py --weights runs/train-seg/exp14/weights/last.pt --data data/ycbv.yaml --camera-topic /camera/color/image_raw --conf-thres 0.9 --iou-thres 0.6

"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

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

from std_msgs.msg import Header
from object_detector_msgs.msg import BoundingBox, Detection, Detections
from object_detector_msgs.srv import detectron2_service_server

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

class YOLOv5:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(
            self,
            weights=ROOT / 'yolov5s.pt',  # model path or triton URL
            source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/fleckerl.yaml',  # dataset.yaml path
            imgsz=(480, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
            retina_masks=True,
            camera_topic='/camera/color/image_raw',
                ):

        # remember some stuff with member variables
        self.augment = augment
        self.img_size = imgsz
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.conf_thres= conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.retina_masks = retina_masks
        self.camera_topic = camera_topic

        source = str(source)
        self.save_img = not nosave and not source.endswith('.txt')  # save inference images

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        self.device = device

        print("\n\n\n")
        print(weights, device, dnn, data, camera_topic)
        print("\n\n\n")

        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        self.model = model
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine

        
        self.auto = self.pt 

        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Half
        self.half = half
        self.half &= (self.pt or self.jit or self.onnx or self.engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if half else self.model.model.float()
        
        bs = 1  # batch_size

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0


        # ROS Stuff
        self.bridge = CvBridge()
        self.pub_detections = rospy.Publisher("/yolov5/detections", Detections, queue_size=10)
        self.service = rospy.Service("/detect_objects", detectron2_service_server, self.service_call)

        self.dt, self.seen = [0.0, 0.0, 0.0, 0.0], 0

    def callback_image(self, msg):
        try:
            img0 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        ros_detections = self.infer(img0) 
        self.pub_detections.publish(ros_detections)   

    def service_call(self, req):
        rgb = req.image
        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480

        try:
            img0 = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            print(e)

        ros_detections = self.infer(img0) 
        return ros_detections

    @smart_inference_mode()
    def infer(self, im0s):

        t1 = time_sync()
        height, width, channels = im0s.shape
        
        #img_size=(480,640)
        stride=32

        img = letterbox(im0s, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        bs = 1  # batch_size

        im = torch.from_numpy(im).to(self.device)
        im = im.s() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        
        t2 = time_sync()
        self.dt[0] += t2 - t1
        
        pred, proto = self.model(im, augment=self.augment)[:2]

        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)

        self.dt[2] += time_sync() - t3

        # Process predictions

        s = ""
        detections = []

        det = pred[0]
        
        self.seen += 1

        im0 = im0s.copy()

        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0s  # for save_crop
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            if self.retina_masks:
                # scale bbox first the crop masks
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                masks = process_mask_native(proto[0], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
            else:
                masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
        
            masks_cpu = masks.cpu().detach().numpy()
            
        
            bbox = det[:, :4].cpu().detach().numpy()
            confidence = det[:, :5].cpu().detach().numpy()
            class_label = det[:, :6].cpu().detach().numpy()

            for obj_id in range(0, confidence.shape[0]):
                detection = Detection()

                # ---
                detection.name = self.names[int(class_label[obj_id][5])]
                # ---

                # ---            
                bbox_msg = BoundingBox()
                bbox_msg.ymin = int(bbox[obj_id][0])
                bbox_msg.xmin = int(bbox[obj_id][1])
                bbox_msg.ymax = int(bbox[obj_id][2])
                bbox_msg.xmax = int(bbox[obj_id][3])
                detection.bbox = bbox_msg
                # ---
                # mask
                # ---
                mask = masks_cpu[obj_id]
                mask_ids = np.argwhere(mask.reshape((height * width)) > 0)
                detection.mask = list(mask_ids.flat)
                # ---

                # ---
                detection.score = confidence[obj_id][4]
                # ---
                # 
                detections.append(detection)      

            # Segments
            segments = [
                scale_segments(im0.shape if self.retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                for x in reversed(masks2segments(masks))]
                
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Mask plotting
            annotator.masks(
                masks,
                colors=[colors(x, True) for x in det[:, 5]],
                im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
                255 if self.retina_masks else im[0])

            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                line = (cls, *seg, conf) if self.save_conf else (cls, *seg)  # label format

                if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                        
        
        ros_detections = Detections()
        ros_detections.width, ros_detections.height = 640, 480
        ros_detections.detections = detections

        # Stream results
        t4 = time_sync()
        self.dt[2] += t4 - t1

        im0 = annotator.result()

        #cv2.imshow("result", im0)
        #cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        rospy.loginfo(f'Inference ({t3 - t2:.3f}s)')
        rospy.loginfo(f'Callback ({t4 - t1:.3f}s)')    

        return ros_detections    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/Objects365.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--camera-topic', type=str, default='/camera/color/image_raw', help='camera topic for input image')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

if __name__ == "__main__":

    try:
        rospy.init_node('yolov5')
        opt = parse_opt()
        check_requirements(exclude=('tensorboard', 'thop'))

        YOLOv5(**vars(opt))

        rospy.spin()
    except rospy.ROSInterruptException:
        pass