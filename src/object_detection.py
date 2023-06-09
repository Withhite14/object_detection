#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from your_package.srv import StreamObjPoseCommand,StreamObjPoseCommandResponse,GetDepthCommand,PinTFCommand,PinTFCommandResponse
from your_package.msg import ObjectPoseCommand
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped
import tf_conversions
from rospkg import RosPack

pack_path = RosPack().get_path('your_package')
weight_path = pack_path + "/weight/hand.weights"
cfg_path = pack_path + "/weight/hand.cfg"
names_path = pack_path + "/weight/hand.names"

class ObjectDetection(object):

    def __init__(self):
        # srv and msg
        self.bridge = CvBridge()
        rospy.init_node("object_detect", anonymous=True)
        rospy.Subscriber("/camera/rgb/image_color", Image, self.update_frame_callback)
        rospy.wait_for_message("/camera/rgb/image_color", Image)
        ########### service #################
        self.depth = rospy.ServiceProxy("/basil/vision/get_depth", GetDepthCommand)
        self.stream_srv=rospy.Service('/basil/vision/stream_obj_pose', StreamObjPoseCommand, self.handle_stream_obj_pose)
        self.stream_srv = False
        self.pinTF_srv = rospy.Service("/basil/vision/pin_tf", PinTFCommand, self.pinTF_callback)
        #####################################
        self.obj_pose_msg=ObjectPoseCommand()
        self.pub=rospy.Publisher('/basil/vision/obj_pose', ObjectPoseCommand, queue_size=10)
        self.image_pub = rospy.Publisher("/annotated_image", Image, queue_size=1)
        # fov camera
        self.h_fov=69.4  #deg
        self.v_fov=42.5  #deg
        #tf2
        self.br=tf2_ros.TransformBroadcaster()
        self.t=TransformStamped()
        self.ls = tf.TransformListener()
        self.static_br = tf2_ros.StaticTransformBroadcaster()
        # network
        self.net = cv2.dnn.readNet(weight_path, cfg_path)
        self.classes = []
        with open(names_path,"r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = [layer_name for layer_name in self.net.getUnconnectedOutLayersNames()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # enable gpu
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.static_tf_name_list = []

    def handle_stream_obj_pose(self,req):
        # Handle requests to the stream_obj_pose service
        self.stream_srv = req.active
        return StreamObjPoseCommandResponse()
        
    def update_frame_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        self.height, self.width, self.channels = self.image.shape
        
    def pinTF_callback(self,data):
        center_x = data.x
        center_y = data.y
        name = data.name
        if name in self.static_tf_name_list:
            count = 1
            new_name = name
            for new_name in self.static_tf_name_list:
                new_name = name + "_new" * count
                count += 1
            name = new_name
        self.static_tf_name_list.append(name)
        pinTF_srv_res = PinTFCommandResponse()
        depth = self.depth([center_x,center_y])
        depth = depth.depth/100 #cm to m
        alpha_rad = (center_x-self.width/2)*(self.h_fov/self.width)*(3.14/180)
        alpha_deg = alpha_rad*(180/3.14)
        beta_rad = (center_y-self.height/2)*(self.v_fov/self.height)*(3.14/180)
        beta_deg = beta_rad*(180/3.14)
        y_px = center_x - (self.width/2)
        z_px = center_y - (self.height/2)
        l_px = y_px/np.sin(alpha_rad)
        dpx = np.sqrt((l_px**2)+(z_px**2))
        ratio = depth/dpx
        x_px = y_px/np.tan(alpha_rad)
        x_meter = x_px*ratio
        y_meter = -(y_px*ratio)
        z_meter = -(z_px*ratio)
        pinTF_srv_res.obj_pixel_x = center_x
        pinTF_srv_res.obj_pixel_y = center_y
        pinTF_srv_res.obj_frame = name 
        pinTF_srv_res.obj_angle_horizontal = alpha_deg
        pinTF_srv_res.obj_angle_vertical = beta_deg
        pinTF_srv_res.obj_meter_x = x_meter
        pinTF_srv_res.obj_meter_y = y_meter
        pinTF_srv_res.obj_meter_z = z_meter
        pinTF_srv_res.obj_meter_depth = depth
        self.static_broadcast_tf(x_meter,y_meter,z_meter,name)
        return pinTF_srv_res
    
    def static_broadcast_tf(self, x, y, z, name):
        self.t.header.stamp = rospy.Time.now()
        self.t.header.frame_id = "camera_link"
        self.t.child_frame_id = name
        self.t.transform.translation.x = x
        self.t.transform.translation.y = y 
        self.t.transform.translation.z = z
        q = tf_conversions.transformations.quaternion_from_euler(0,0,0)
        self.t.transform.rotation.x = q[0]
        self.t.transform.rotation.y = q[1]
        self.t.transform.rotation.z = q[2]
        self.t.transform.rotation.w = q[3]
        self.static_br.sendTransform(self.t)
        try:
            # listen
            self.ls.waitForTransform("map",name,rospy.Time(0),rospy.Duration(4.0))
            trans_obj_map,rot_obj_map = self.ls.lookupTransform("map",name,rospy.Time(0))
            
            # tf from obj_name to map
            self.t.header.stamp = rospy.Time.now()
            self.t.header.frame_id = "map"
            self.t.child_frame_id = name 
            self.t.transform.translation.x = trans_obj_map[0]
            self.t.transform.translation.y = trans_obj_map[1]
            self.t.transform.translation.z = trans_obj_map[2]
            self.t.transform.rotation.x = rot_obj_map[0]
            self.t.transform.rotation.y = rot_obj_map[1]
            self.t.transform.rotation.z = rot_obj_map[2]
            self.t.transform.rotation.w = rot_obj_map[3]
            self.static_br.sendTransform(self.t)
        except Exception as err:
            print(err)
            print("tf error")
            pass

    def broadcast_tf(self, x, y, z, obj_name):
        self.t.header.stamp = rospy.Time.now()
        self.t.header.frame_id = "camera_link"
        self.t.child_frame_id = obj_name
        self.t.transform.translation.x = x
        self.t.transform.translation.y = y 
        self.t.transform.translation.z = z
        q = tf_conversions.transformations.quaternion_from_euler(0,0,0)
        self.t.transform.rotation.x = q[0]
        self.t.transform.rotation.y = q[1]
        self.t.transform.rotation.z = q[2]
        self.t.transform.rotation.w = q[3]
        self.br.sendTransform(self.t)
        try:
            # listen
            self.ls.waitForTransform("map",obj_name,rospy.Time(0),rospy.Duration(4.0))
            trans_obj_map,rot_obj_map = self.ls.lookupTransform("map",obj_name,rospy.Time(0))
            
            # tf from obj_name to map
            self.t.header.stamp = rospy.Time.now()
            self.t.header.frame_id = "map"
            self.t.child_frame_id = obj_name 
            self.t.transform.translation.x = trans_obj_map[0]
            self.t.transform.translation.y = trans_obj_map[1]
            self.t.transform.translation.z = trans_obj_map[2]
            self.t.transform.rotation.x = rot_obj_map[0]
            self.t.transform.rotation.y = rot_obj_map[1]
            self.t.transform.rotation.z = rot_obj_map[2]
            self.t.transform.rotation.w = rot_obj_map[3]
            self.br.sendTransform(self.t)
        except Exception as err:
            print(err)
            print("tf error")
            pass
              
    def pub_image(self,img):
        #pub img
        annotated_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.image_pub.publish(annotated_img)

    def main(self):    
        height, width, channels = self.image.shape
        font = cv2.FONT_HERSHEY_PLAIN
        print(f"height: {height}")
        print(f"width: {width}")
        rate = rospy.Rate(5)
        ###################### latest list #########################################
        x_latest = []
        y_latest = []
        angle_h_latest = []
        angle_v_latest = []
        x_meter_latest = []
        y_meter_latest = []
        z_meter_latest = []
        obj_px_left_x_latest = []
        obj_px_top_y_latest = []
        obj_px_w_latest =  []
        obj_px_h_latest = []
        obj_frames_latest = []
        obj_names_latest = []
        obj_meter_depth_latest = []
        latest_br = {}
        ###################################################################
        while not rospy.is_shutdown():
            image = self.image
        ############## pub list ##########################################
            x_pub = []
            y_pub = []
            obj_names = []
            angle_h_pub = []
            angle_v_pub = []
            x_meter_pub = []
            y_meter_pub = []
            z_meter_pub = []
            obj_px_left_x = []
            obj_px_top_y = []
            obj_px_w =  []
            obj_px_h = []
            obj_frames = []
            obj_counts = {}
            obj_meter_depth = []
        ##################################################################
            if self.stream_srv:
                start_time_fps = rospy.get_rostime()
               
                # detect
                blob = cv2.dnn.blobFromImage(image, 1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
                self.net.setInput(blob)
                outputs = self.net.forward(self.output_layers)
                boxes = []
                confs = []
                class_ids = []
                ###################################################################
                ################ detect ##########################################
                for output in outputs:
                    for detect in output:
                        scores = detect[5:]
                        class_id = np.argmax(scores)
                        conf = scores[class_id]
                        if conf > 0.4:
                            center_x = int(detect[0] * width)
                            center_y = int(detect[1] * height)
                            w = int(detect[2] * width)
                            h = int(detect[3] * height)
                            x = int(center_x - w/2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confs.append(float(conf))
                            class_ids.append(class_id)
                ###################################################################

                ########################### only one object #######################                        
                indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
                obj_counts = {}
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(self.classes[class_ids[i]])
                        conf = confs[i]
                        if label not in obj_counts:
                            obj_counts[label] = 1
                        else:
                            obj_counts[label] += 1
                        
                        ########## calculate x,y,z ###############################
                        center_x = int(x+w/2)
                        center_y = int(y+h/2)
                        depth = self.depth([center_x,center_y])
                        depth = depth.depth/100 #cm to m
                        alpha_rad = (center_x-width/2)*(self.h_fov/width)*(3.14/180)
                        alpha_deg = alpha_rad*(180/3.14)
                        beta_rad = (center_y-height/2)*(self.v_fov/height)*(3.14/180)
                        beta_deg = beta_rad*(180/3.14)
                        y_px = center_x - (width/2)
                        z_px = center_y - (height/2)
                        l_px = y_px/np.sin(alpha_rad)

                        dpx = np.sqrt((l_px**2)+(z_px**2))
                        ratio = depth/dpx
                        x_px = y_px/np.tan(alpha_rad)
                        x_meter = x_px*ratio
                        y_meter = -(y_px*ratio)
                        z_meter = -(z_px*ratio)
                        ########################################################
                        ######## append to list#################################
                        x_pub.append(int(center_x))
                        y_pub.append(int(center_y))
                        obj_names.append(label)
                        obj_frames.append(label + '_' + str(obj_counts[label]))
                        angle_h_pub.append(float(alpha_deg))
                        angle_v_pub.append(float(beta_deg))
                        x_meter_pub.append(float(x_meter))
                        y_meter_pub.append(float(y_meter))
                        z_meter_pub.append(float(z_meter))
                        obj_px_left_x.append(x)
                        obj_px_top_y.append(y)
                        obj_px_w.append(w)
                        obj_px_h.append(h)
                        obj_meter_depth.append(depth)
                        ##############################
                        #draw bounding box
                        color = self.colors[class_ids[i]]
                        image = cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)
                        image = cv2.putText(image,  label + '_' + str(obj_counts[label]) + " " + str(round(conf, 2)), (x, y - 5), font, 1, color, 1)                        ############################# pub and broadcast ###################
                        self.obj_pose_msg.obj_pixel_x = x_pub
                        self.obj_pose_msg.obj_pixel_y = y_pub
                        self.obj_pose_msg.obj_name = obj_names
                        self.obj_pose_msg.obj_frame = obj_frames
                        self.obj_pose_msg.obj_angle_horizontal = angle_h_pub
                        self.obj_pose_msg.obj_angle_vertical = angle_v_pub
                        self.obj_pose_msg.obj_meter_x = x_meter_pub
                        self.obj_pose_msg.obj_meter_y = y_meter_pub
                        self.obj_pose_msg.obj_meter_z = z_meter_pub
                        self.obj_pose_msg.obj_pixel_left_x = obj_px_left_x
                        self.obj_pose_msg.obj_pixel_top_y = obj_px_top_y
                        self.obj_pose_msg.obj_pixel_width = obj_px_w
                        self.obj_pose_msg.obj_pixel_height = obj_px_h
                        self.obj_pose_msg.obj_meter_depth = obj_meter_depth
                        rospy.loginfo("Detected!!")

                self.pub.publish(self.obj_pose_msg)
                x_meter_latest = x_meter_pub
                y_meter_latest = y_meter_pub
                z_meter_latest = z_meter_pub
                obj_frames_latest = obj_frames
                obj_names_latest = obj_names
                obj_px_h_latest = obj_px_h
                obj_px_w_latest = obj_px_w
                obj_px_left_x_latest = obj_px_left_x
                obj_px_top_y_latest = obj_px_top_y
                angle_h_latest = angle_h_pub
                angle_v_latest = angle_v_pub
                x_latest = x_pub
                y_latest = y_pub
                obj_meter_depth_latest = obj_meter_depth
                for i in range(len(obj_frames)):
                    self.broadcast_tf(x_meter_pub[i], y_meter_pub[i], z_meter_pub[i], obj_frames[i])
                    latest_br[obj_frames[i]] = {'x': x_meter_pub[i],'y':y_meter_pub[i],'z':z_meter_pub[i]}
                fps = 1/(rospy.get_rostime() - start_time_fps).to_sec()
                image = cv2.putText(image,f"fps : {fps:.3f}",(10,30),font,2,(0,0,255),2)

                        ###################################################################
                
            ######################### pub and broadcast latest #############################
            else :
                self.obj_pose_msg.obj_pixel_x = x_latest
                self.obj_pose_msg.obj_pixel_y = y_latest
                self.obj_pose_msg.obj_name = obj_names_latest
                self.obj_pose_msg.obj_frame = obj_frames_latest
                self.obj_pose_msg.obj_angle_horizontal = angle_h_latest
                self.obj_pose_msg.obj_angle_vertical = angle_v_latest
                self.obj_pose_msg.obj_meter_x = x_meter_latest
                self.obj_pose_msg.obj_meter_y = y_meter_latest
                self.obj_pose_msg.obj_meter_z = z_meter_latest
                self.obj_pose_msg.obj_pixel_left_x = obj_px_left_x_latest
                self.obj_pose_msg.obj_pixel_top_y = obj_px_top_y_latest
                self.obj_pose_msg.obj_pixel_width = obj_px_w_latest
                self.obj_pose_msg.obj_pixel_height = obj_px_h_latest
                self.obj_pose_msg.obj_meter_depth = obj_meter_depth_latest
                self.pub.publish(self.obj_pose_msg)
                for obj_frame in latest_br:
                    x_meter=latest_br[obj_frame]['x']
                    y_meter=latest_br[obj_frame]['y']
                    z_meter=latest_br[obj_frame]['z']
                    self.broadcast_tf(x_meter,y_meter,z_meter,obj_frame)
                rate.sleep()
            ################################################################################
            
            # publish image
            self.pub_image(image)
              

if __name__ == "__main__":
    try:
        obj = ObjectDetection()
        obj.main()
    except rospy.ROSInterruptException:
        pass
