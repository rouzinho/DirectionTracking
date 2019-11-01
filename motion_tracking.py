import pyrealsense2 as rs
from collections import deque
from imutils.video import VideoStream
import argparse
import datetime
import numpy as np
import cv2
import dlib
import time
import imutils
from numpy import linalg as la
import rospy
import sys
import csv
import math

from std_msgs.msg import Float64

pitch = Float64(0.0)
yaw = Float64(0.0)

def movePitch(p,y):
    #tmp_p = Float64(-p.data)
    #tmp_y = Float64(-y.data)
    head_pitch.publish(p)
    head_yaw.publish(y)

def calcVector(x_center,y_center,x_face,y_face,p1,y1):
    x = x_face - x_center
    y = y_face - y_center
    #x = x_center - x_face
    #y = y_center - y_face
    tmp = [x,y]
    #print("magnitude vector :")
    #print(la.norm(tmp))
    if la.norm(tmp) > 20:
        vect = tmp/la.norm(tmp)
        vect = vect/20
        p = vect[0]
        y = vect[1]
        #print("translation vector :")
        #print(vect)
        #p1 = Float64(p+p1.data)
        #y1 = Float64(y+y1.data)
        if p < 0:
            p1 = Float64(p1.data+0.01)
        else :
            p1 = Float64(p1.data-0.01)
        if y > 0:
            y1 = Float64(y1.data+0.01)
        else :
            y1 = Float64(y1.data-0.01)


    return [p1,y1]

max_value = 2

def scale_input(value):
    global max_value
    if value > max_value:
        max_value = value
    return (value/max_value) * 2

#rospy.init_node('gummi', anonymous=True)
#r = rospy.Rate(30)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-b", "--buffer", type=int, default=32,help="max buffer size")
args = vars(ap.parse_args())

#define white tracking
lower_black = (0,0,0)
upper_black = (180,255,30)
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	#vs = VideoStream(src=0).start()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    OUTPUT_SIZE_WIDTH = 775
    OUTPUT_SIZE_HEIGHT = 600
    #rectangleColor = (0,165,255)
    # Start streaming
    pipeline.start(config)
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None
#motion_detector = rospy.Publisher("/motion_detector", Float64, queue_size=10)

try:
    while True :

        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        #baseImage = cv2.resize( color_image, (640, 480))
        #gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if color_image is None:
            break


        baseImage = imutils.resize(color_image, width=600)
        height, width, channels = baseImage.shape
        print(height)
        print(width)
        #baseImage = cv2.resize( color_image, (640, 320))
        blurred = cv2.GaussianBlur(baseImage, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, lower_black, upper_black)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    		# only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(baseImage, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(baseImage, center, 5, (0, 0, 255), -1)

        pts.appendleft(center)

        tmp = calcVector(300,225,center[0],center[1],pitch,yaw)
        pitch = tmp[0]
        yaw = tmp[1]
        movePitch(pitch,yaw)

        #for i in range(1, len(pts)):
        #    # if either of the tracked points are None, ignore
            # them
        #    if pts[i - 1] is None or pts[i] is None:
        #        continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
        #    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #    cv2.line(baseImage, pts[i - 1], pts[i], (0, 0, 255), thickness)

        #images = np.hstack((color_image, largeResult))
        cv2.namedWindow('BaseImage', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Threshold', cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow('FrameDelta', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('BaseImage', baseImage)
        cv2.imshow('Threshold', mask)
        #cv2.imshow('FrameDelta', frameDelta)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:

    # Stop streaming
    pipeline.stop()
