#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from your_package.srv import GetDepthCommand
from cv_bridge import CvBridge

#################### ERROR CODE ###########################
# -10 : Lenght of position in service caller is not 2.
# -11 : Position data is out of bound.
# -12 : Depth of position that you call lower than bound.
###########################################################

######################### MANUAL ################################
# Check image size in : self.img_size is (720,1280) by default.
# Set roi size in : self.roi_size is 4 by default.
#################################################################


class GetDepth(object):
    def __init__(self):
        
        ########## (BASIC SETTING) ##########
        self.roi_size = 4
        self.img_size = (480,640)
        #####################################

        self.bridge = CvBridge()
        rospy.init_node("get_depth",anonymous=True)
        self.active = False
        self.shutdown = False
        self.lower_depth_bound = 40 #cm REF from Min-Z in document link: https://www.intelrealsense.com/depth-camera-d415/
        self.min_moving_avg_count = 3
        self.img_sub = rospy.Subscriber("/camera/depth_registered/image",Image,self.callback,queue_size=10)
        self.get_depth_ser = rospy.Service("/basil/vision/get_depth",GetDepthCommand,self.get_depth)
        self.position = None
        self.ans = []

        rospy.loginfo("GetDepth node is started!")

    def callback(self,data):
        if not self.active or (self.position is None):
            return 1
        data = self.bridge.imgmsg_to_cv2(data,desired_encoding='passthrough')
        depth_array = np.array(data, dtype=np.float32)
        if depth_array is None or self.position is None:
            return 1
        react = depth_array[self.position[0]-self.roi_size:self.position[0]+self.roi_size , self.position[1]-self.roi_size:self.position[1]+self.roi_size]
        print(react.mean()*100)
        self.ans.append(react.mean()*100)

    def get_depth(self,data):
        if len(data.position) != 2:
            rospy.logerr("Position error!")
            return -10 # Request error
        data.position = [data.position[1],data.position[0]]
        if data.position[0] > self.img_size[0]-self.roi_size or data.position[0] < self.roi_size or data.position[1] > self.img_size[1]-self.roi_size or data.position[1] < self.roi_size:
            rospy.logerr("Position out of bound!")
            return -11
        self.position = data.position
        self.active = True
        ans = self.ans
        while len(self.ans) < self.min_moving_avg_count or len(ans) < self.min_moving_avg_count:
            ans = self.ans
        ans = np.array(ans)
        ans = ans.mean()
        if ans < self.lower_depth_bound:
            self.ans = []
            self.position = None
            self.active = False
            rospy.logerr("Depth out of bound!")
            return -12
        self.ans = []
        self.position = None
        self.active = False
        print("Depth : ",ans)
        return ans

if __name__ == "__main__":
    d = GetDepth()
    
    while not rospy.is_shutdown() and not d.shutdown:
        pass

    cv2.destroyAllWindows()
