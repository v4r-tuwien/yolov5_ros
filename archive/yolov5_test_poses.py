#! /usr/bin/env python3
import rospy
from object_detector_msgs.msg import GenericImgProcAnnotatorAction, GenericImgProcAnnotatorGoal, PoseStWithConfidence
from sensor_msgs.msg import Image
from object_detector_msgs.srv import detectron2_service_server, estimate_poses

def detect(rgb):
    rospy.wait_for_service('detect_objects_yolov5')
    try:
        detect_objects = rospy.ServiceProxy('detect_objects_yolov5', detectron2_service_server)
        response = detect_objects(rgb)
        return response.detections.detections
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def estimate_pose(detections, rgb, depth):
    rospy.wait_for_service('estimate_pose_gdrn')
    try:
        estimate = rospy.ServiceProxy('estimate_pose_gdrn', estimate_poses)
        response = estimate(detections, rgb, depth)
        return response.poses
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def get_poses():
    print('Waiting for images')
    rgb = rospy.wait_for_message('/camera/color/image_raw', Image)
    depth = rospy.wait_for_message('/camera/depth/image_rect_raw', Image)
    print('Detection Service ...')
    detections = detect(rgb)
    print(detections)
    print('Detection completed, now Pose Estimation ...')
    if len(detections) > 0:
        poses = estimate_pose(detections[0], rgb, depth)
        print(poses)
        print('Got the poses, now from the beginning...')
    else:
        print("no pose ...")
        
if __name__ == "__main__":
    rospy.init_node("get_poses")
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        get_poses()
        r.sleep()

    rospy.spin()