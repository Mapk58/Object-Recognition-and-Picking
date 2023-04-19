#!/usr/bin/env python
from __future__ import print_function

import sys
import json
import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from object_recognition import ObjectRecognition


class image_converter:

    def __init__(self):
        self.depth_image_pub = rospy.Publisher("/project/depth_map", Image, queue_size=-1)
        self.contours_depth_pub = rospy.Publisher("/project/contours_depth", Image, queue_size=-1)
        self.contours_yolo_pub = rospy.Publisher("/project/contours_yolo", Image, queue_size=-1)
        self.contours_united_pub = rospy.Publisher("/project/contours_united", Image, queue_size=-1)
        self.objects_pub = rospy.Publisher("project/objects", String, queue_size=-1)

        self.recognition = ObjectRecognition(10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_rect_color", Image, self.callback_rgb)
        self.depth_sub = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image, self.callback_dpth)
        self.camera_info_sub = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.callback_info)

    def callback_info(self, data):
        try:
            if self.recognition.camera_info is None:
                self.recognition.camera_info = data
        except CvBridgeError as e:
            print(e)

    def callback_rgb(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.recognition.updateImage(cv_image)
        except CvBridgeError as e:
            print(e)

        contours_depth = self.recognition.getObjectsByDepth()
        contours_yolo = self.recognition.classifyObjects()
        contours_united = self.recognition.overlayContours(contours_depth, contours_yolo)    

        json_string = json.dumps({"Data:":self.recognition.objects}, indent=1)
        print(json_string)
        print('========')

        # Example of publishing
        try:
            self.depth_image_pub.publish(
                self.bridge.cv2_to_imgmsg((self.recognition.fixed_depth / np.max(self.recognition.fixed_depth) * 255).astype(np.uint8), "mono8"))
            self.contours_depth_pub.publish(
                self.bridge.cv2_to_imgmsg(self.recognition.drawContours(contours_depth), "bgr8"))
            self.contours_yolo_pub.publish(
                self.bridge.cv2_to_imgmsg(self.recognition.drawRectangles(contours_yolo), "bgr8"))
            self.contours_united_pub.publish(
                self.bridge.cv2_to_imgmsg(self.recognition.drawContours(contours_united), "bgr8"))
            self.objects_pub.publish("Data:" + str(self.recognition.objects))
        except CvBridgeError as e:
            print(e)

    # depth callback
    def callback_dpth(self, data):
        try:
            cv_dpth = self.bridge.imgmsg_to_cv2(data, "16UC1")
            self.recognition.updateDepth(cv_dpth)
        except CvBridgeError as e:
            print(e)


def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
    print("Node 'manipulator' started successfully!\nPlease make sure there are no foreign objects on the table when starting.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
