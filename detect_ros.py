# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
#from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.get_ros_image import LoadROSImage
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
            camera_topic='/camera/color/image_raw',
                ):

        # remember some stuff with member variables
        self.augment = augment
        self.img_size = imgsz
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.view_img = view_img
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.conf_thres= conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.camera_topic = camera_topic

        source = str(source)
        self.save_img = not nosave and not source.endswith('.txt')  # save inference images
        '''
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download
        '''

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
        
        
        #vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0


        # ROS Stuff
        self.bridge = CvBridge()

        self.pub_bounding_box = rospy.Publisher("/camera/color/bounding_box", RegionOfInterest, queue_size=10)
        self.pub_cropped_img = rospy.Publisher("/camera/color/cropped_img", Image, queue_size=10)

        self.pub_detection2array = rospy.Publisher("/detection2d", Detection2DArray, queue_size=10)

        self.sub = rospy.Subscriber(camera_topic, Image, self.callback_image)

        self.dt, self.seen = [0.0, 0.0, 0.0, 0.0], 0

    def callback_image(self, msg):
        try:
            img0 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        #img0 = cv2.transpose(img0)

        self.infer(img0)    

    @smart_inference_mode()
    def infer(self, im0s):

        t1 = time_sync()
        height, width, channels = im0s.shape
        
        img_size=(480,640)
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

        pred = self.model(im, augment=self.augment, visualize=False)

        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        self.dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        s = ""

        for i, det in enumerate(pred):  # per image
            self.seen += 1

            im0 = im0s.copy()

            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0s  # for save_crop
            annotator = Annotator(im0, line_width=3, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Publish the Bounding Box Info
                bb = det[:, :4].cpu().detach().numpy()
                det_cpu = det.cpu().detach().numpy()
                print(det_cpu)
                print()
                print(bb)
                print(len(bb))
                print()

                # Detection2D Messages in a loop
                #
                detection2darray_msg = Detection2DArray()
                for detection in det_cpu:
                    detection2d_msg = Detection2D()

                    object_hypothesis = ObjectHypothesisWithPose()
                    object_hypothesis.id = int(detection[5])
                    object_hypothesis.score = detection[4]
                    detection2d_msg.results.append(object_hypothesis)

                    detection2d_msg.bbox.size_x = detection[3]-detection[1]
                    detection2d_msg.bbox.size_y = detection[2]-detection[0]

                    detection2d_msg.bbox.center.x = detection[0] + (detection[2] - detection[0])/2
                    detection2d_msg.bbox.center.y = detection[1] + (detection[3] - detection[1])/2

                    detection2darray_msg.detections.append(detection2d_msg)

                self.pub_detection2array.publish(detection2darray_msg)

                # Bounding Box only Message publish
                #
                bb = bb[0].astype(int)
                bb_msg = RegionOfInterest()
                bb_msg.x_offset = bb[0]
                bb_msg.y_offset = bb[1]
                bb_msg.height = bb[3]
                bb_msg.width = bb[2]

                self.pub_bounding_box.publish(bb_msg)

                # Publish the Cropped Image
                #           x:      x+w         y:      y+h
                #           bb[1]:  bb[3]       bb[0]:  bb[2]
                crop = imc[bb[1]:bb[3], bb[0]:bb[2]]
                cropped_img_msg = self.bridge.cv2_to_imgmsg(crop, encoding="passthrough")
                self.pub_cropped_img.publish(cropped_img_msg)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format

                    if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))


        # Stream results
        t4 = time_sync()
        self.dt[2] += t4 - t1

        im0 = annotator.result()

        cv2.imshow("result", im0)
        cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        rospy.loginfo(f'Inference ({t3 - t2:.3f}s)')
        rospy.loginfo(f'Callback ({t4 - t1:.3f}s)')        

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
