# Object-Recognition-and-Picking
This project is a part of Master's Computer Vision course at Innopolis University

---

## Idea: 
Our project is a set of methods and algorithms for working with a manipulator, on which an RGBD camera and a grip are attached, which allows you to capture objects.
In this project we are focusing on the following tasks: 
1. **Table Position Calibration :** Since the table under the camera may have a certain angle, it is necessary to automatically determine the angle of inclination and other parameters. 
2. **Object detection :** Using a depth map and a color image allows you to determine the number and size of objects, as well as their position and orientation relative to the table coordinate system.
3. **Object recognition:** Using pre-trained YOLO will allow us to recognize objects on the table, which will make it possible to give human-readable commands to the manipulator.

---

## Time:
|Weeks |Task                               |
|---	|---	                                |
|   1	|Installation of the required drivers |
|   1	|Table Calibration   	                |
|   2	|Object Detection         	          |
|   2 |Object Recognition                   |

## Requirements:
* ros noetic
* python:
* ultralytics==8.0.20
* pyrealsense2
* opencv-python

## Results:
System recieve video stream and detect edges and depth map:
![1](https://user-images.githubusercontent.com/45263316/233270241-4a0d70b5-93ae-42e2-874c-a2c8d343036b.png)
As an output, logs are sent to the console:
~~~
{
 "Data:": [
  {
   "Name": "apple",
   "Bounding_box": [
    267.0,
    194.0,
    505.0,
    423.0
   ],
   "Center_in_pixels": [
    386,
    308
   ],
   "Height_in_cm": 5.4,
   "Real_coords": [
    246.1999969482422,
    70.09951782226562,
    15.287549018859863
   ]
  },
  {
   "Name": "apple",
   "Bounding_box": [
    676.0,
    260.0,
    940.0,
    518.0
   ],
   "Center_in_pixels": [
    808,
    389
   ],
   "Height_in_cm": 6.5,
   "Real_coords": [
    243.0,
    -42.28042221069336,
    -6.351202011108398
   ]
  }
 ]
}
~~~
## Contributors:

- [@mapk58](https://github.com/mapk58)
- [@olegogogo](https://github.com/olegogogo)
