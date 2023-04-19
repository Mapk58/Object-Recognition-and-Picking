import numpy as np
import cv2 as cv
import pyrealsense2
from ultralytics import YOLO
import random


class ObjectRecognition:
    def __init__(self, queue_size) -> None:
        self.net = YOLO('yolov8s.pt')
        self.image = None
        self.gray = None
        self.depth = None
        self.depths = np.array([])
        self.fixed_depth = None
        self.speed = np.array([0.])
        self.queue_size = queue_size
        self.camera_info = None
        self.table = None
        self.objects = None

    def updateImage(self, frame):
        self.image = frame
        self.gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def updateDepth(self, frame):
        self.depth = frame
        self.depths = np.append(self.depths, [self.depth], axis=0) if len(
            self.depths) > 0 else np.array([self.depth])
        if len(self.depths) > self.queue_size:
            self.depths = self.depths[1:]
        self.fixed_depth = np.mean(self.depths, axis=0).astype(np.uint16)
        self.mask_depth = self.fixed_depth.copy()
        self.mask_depth[self.mask_depth>0] = 255
        self.mask_depth = self.mask_depth.astype(np.uint8)
        
        if self.table is None:
            self.calibTable()
    
    def getObjectsByDepth(self):
        if self.table is None:
            print("Table was not calibrated yet! Please wait.")
            return []
        objects_depth = self.table.astype(int) - self.fixed_depth.astype(int)
        objects_depth = objects_depth / np.max(objects_depth) * 255
        objects_depth[objects_depth < 10] = 0
        objects_depth[objects_depth > 10] = 255
        objects_depth = cv.morphologyEx(
            objects_depth, cv.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=3)
        
        contours, hierarchy = cv.findContours(
            objects_depth.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # cv.imshow('Objects detected', objects_depth.astype(np.uint8))

        return contours
        
    def _genPlane(self, a, b, c, d):
        return np.array(np.linspace(np.linspace(a, b, 1280), np.linspace(c, d, 1280), 720)).astype(np.uint8)

    def _evalPlane(self, img, plane):
        diff = img.astype(int) - plane.astype(int)
        diff = np.abs(diff)
        diff *= self.mask_depth
        res = np.sum(diff)/np.sum(self.mask_depth)
        return res

    def calibTable(self, accurate=False):
        max_value = np.max(self.fixed_depth)

        image = (self.fixed_depth/max_value*255).astype(np.uint8)
        
        q,w,e,r = 0,0,0,0
        res = 9999999999
        
        dist = 200
        c1 = np.unique(image[:dist, :dist])
        c2 = np.unique(image[:dist, -dist:])
        c3 = np.unique(image[-dist:, :dist])
        c4 = np.unique(image[-dist:, -dist:])

        for i in range(5000 if accurate else 400):
            if accurate:
                a,b,c,d = random.randint(0, 256), random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)
            else:
                a,b,c,d = random.choice(c1), random.choice(c2), random.choice(c3), random.choice(c4)
            
            res_now = self._evalPlane(self.fixed_depth, self._genPlane(a,b,c,d)/255*max_value)
            if res_now < res:
                q,w,e,r = a,b,c,d
                res = res_now
        
        # print(res/self._evalPlane(self.fixed_depth, self._genPlane(230,250,150,160)/255*max_value))

        self.table = (self._genPlane(q,w,e,r)/255*max_value).astype(np.uint16)

    def classifyObjects(self):
        result = self.net(self.image)

        classes = result[0].boxes.cls
        coords = result[0].boxes.xyxy
        classes = [self.net.names[int(cls)] for cls in classes]
        coords = [crd.tolist() for crd in coords]
        yolo_centers = [[int((i[0]+i[2])/2),int((i[1]+i[3])/2)] for i in coords]
        yolo_real_coords = [self.getRealCoords(i[0],i[1]) for i in yolo_centers]

        if self.table is None:
            pass    
        else:
            yolo_heights = [int(self.table[i[1],i[0]]) - self.fixed_depth[i[1],i[0]]  for i in yolo_centers]
            objects = []
            for i in range(len(classes)):
                obj = {
                    'Name': classes[i],
                    'Bounding_box': coords[i],
                    'Center_in_pixels': yolo_centers[i],
                    'Height_in_cm': yolo_heights[i]/10,
                    'Real_coords': yolo_real_coords[i],
                    }
                objects.append(obj)
            self.objects = objects
        return coords

    def drawContours(self, contours):
        frame = self.image.copy()
        for contour in contours:
            hull = cv.convexHull(contour)
            cv.drawContours(frame, [hull], 0, (0, 0, 255), 3)
        return frame
    
    def drawRectangles(self, squares):
        frame = self.image.copy()
        for i in squares:
            frame = cv.rectangle(frame, [int(i[0]), int(i[1])], [int(i[2]), int(i[3])], (0,0,255), 3)
        return frame

    def overlayContours(self, contours, squares):

        buf_1 = np.zeros_like(self.depth).astype(np.uint8)
        buf_2 = np.zeros_like(self.depth).astype(np.uint8)

        for contour in contours:
            hull = cv.convexHull(contour)
            cv.drawContours(buf_1, [hull], 0, 255, -1)
        for i in squares:
            buf_2 = cv.rectangle(buf_2, [int(i[0]), int(i[1])], [int(i[2]), int(i[3])], 255, -1)

        overlayed_image = cv.bitwise_and(buf_1, buf_2)
        overlayed_image = cv.morphologyEx(
            overlayed_image, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        contours, hierarchy = cv.findContours(
            overlayed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours

    def getRealCoords(self, x, y):
        depth = np.mean(self.depths, axis=0)[y][x]
        # depth = self.depth[y][x]

        cameraInfo = self.camera_info
        _intrinsics = pyrealsense2.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.K[2]
        _intrinsics.ppy = cameraInfo.K[5]
        _intrinsics.fx = cameraInfo.K[0]
        _intrinsics.fy = cameraInfo.K[4]
        _intrinsics.model = pyrealsense2.distortion.none
        _intrinsics.coeffs = [i for i in cameraInfo.D]
        result = pyrealsense2.rs2_deproject_pixel_to_point(
            _intrinsics, [x, y], depth)
        return [result[2], -result[0], -result[1]]

    