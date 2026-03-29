#!/usr/bin/env python3
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import os
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from tf.transformations import quaternion_from_euler
import math
from collections import namedtuple
import monte_carlo_pack as mcp
import rospkg

class DronePicCollectNode:
    def __init__(self, data_path="sign_pics/", pics_per_sign=20, ellipse_width=1.5, ellipse_height=1, distance_from_sign=0.5):
        """
            Initializes node to fly to each sign and collect photos
            
            @params data_path - path to folder into which photos should be stored
        """
        rospy.init_node('drone_pic_collector', anonymous=True)

        # get constructor args from ROS param service
        raw_path = rospy.get_param('~data_path', "sign_pics")
        self.PICS_PER_SIGN = rospy.get_param('~pics_per_sign', 20)
        self.ELLIPSE_W = rospy.get_param('~ellipse_width', 1.5)
        self.ELLIPSE_H = rospy.get_param('~ellipse_height', 1)
        self.PERP_DIST_FROM_SIGN = rospy.get_param('~distance_from_sign', 0.5)

        # get package path using Roses package management
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('drone_pic_collector')


        # if user provided relative path, join to package path
        if not raw_path.startswith('/') and not raw_path.startswith('~'):
            self.DATA_PATH = os.path.join(package_path, raw_path)
        else:
            # handle home directory or absolute paths
            self.DATA_PATH = os.path.expanduser(raw_path)
    
        if not os.path.exists(self.DATA_PATH):
            try:
                print(f"Attempting to create directory: {self.DATA_PATH}")
                os.makedirs(self.DATA_PATH)
                rospy.loginfo(f"Created directory: {self.DATA_PATH}")
            except OSError as e:
                rospy.logerr(f"Could not create directory {self.DATA_PATH}: {e}")


        self.ns = rospy.get_namespace().strip('/')
        self.bridge = CvBridge()
        self.SPEED = 10
        
        self.MODEL_NAME = self.ns

        
        self.take_pic = False
        self.current_sign = 1
        self.pic_num = 0

        self.camera_topic = "camera1/image_raw"

        # self.image_sub = rospy.Subscriber("camera1/image_raw", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)

        Pose = namedtuple('Pose', ['x', 'y', 'z', 'yaw'])

        # absolute positions of each sign x (m), y (m), z (m), yaw (degrees)
        abs_sign_dict = {
            1: Pose( 5.81,  1.64,  0.1,  270),
            2: Pose( 5.16, -1.35,  0.1,  270),
            3: Pose( 4.00, -1.67,  0.1,  180),
            4: Pose( 0.83, -0.54,  0.1,   90),
            5: Pose( 0.83,  1.50,  0.1,  270),
            6: Pose(-3.41,  1.71,  0.1,  180),
            7: Pose(-3.80, -2.01,  0.1,    0)
        }

        # create locations that offset the drone from the sign 
        self.sign_dict = abs_sign_dict
        for i in range(1, 8):
            old_pose = abs_sign_dict[i]
            yaw_rad = math.radians(old_pose.yaw)
            offset_x = old_pose.x - self.PERP_DIST_FROM_SIGN * math.cos(yaw_rad)
            offset_y = old_pose.y - self.PERP_DIST_FROM_SIGN * math.sin(yaw_rad)

            self.sign_dict[i] = Pose(offset_x, offset_y, old_pose.z, old_pose.yaw)
        
        rospy.sleep(1.0) # sleep for 1s to let nodes initialize
        
        # GENERATE RELATIVE COORDINATES FOR EACH PHOTO AT EACH SIGN
        self.mcp = mcp.MonteCarloPack()
        # generates list of horizontal and z coordinates for robot to position at for each photo
        self.stops = self.mcp.get_point_list(self.ELLIPSE_W, self.ELLIPSE_H, self.PICS_PER_SIGN, 15)


    # def image_callback(self, data):
    #     if not self.take_pic:
    #         return
    #     rospy.loginfo(f"Collecting photos at sign {self.current_sign}...")
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #         self.pic_num += self.pic_num
    #         suffix = "{:04d}".format(self.pic_num)
    #         cv.imwrite(os.path.join(self.DATA_PATH, f"sign{self.current_sign}_{suffix}.png"), cv_image)
    #     except CvBridgeError as e:
    #         rospy.logerr(f"CvBridge Error: {e}")
    
    def fly_to_pos(self, x, y, z, yaw):
        """
            Teleports drone to coordinates and orientation using gazebo set model state service.
        """
        while not rospy.is_shutdown():
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                yaw_rad = math.radians(yaw)

                # convert Euler (roll, pitch, yaw) to quaternion
                q = quaternion_from_euler(0,0,yaw_rad)

                apply_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                
                state = ModelState()
                state.model_name = self.MODEL_NAME

                state.pose.position.x = x
                state.pose.position.y = y
                state.pose.position.z = z

                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]

                state.twist.linear.x = 0
                state.twist.linear.y = 0
                state.twist.linear.z = 0

                state.twist.angular.x = 0
                state.twist.angular.y = 0
                state.twist.angular.z = 0

                state.reference_frame = "world"

                response = apply_state(state)
            except rospy.ServiceException as e:
                print(f"Service call failed: {e}")
            return
    
    def fly_to_sign(self, sign_num):
        x = self.sign_dict[sign_num].x
        y = self.sign_dict[sign_num].y
        z = self.sign_dict[sign_num].z
        yaw = self.sign_dict[sign_num].yaw
        self.fly_to_pos(x, y, z, yaw)
        return
    
    def collect_photos(self):
        while not rospy.is_shutdown():
            if self.current_sign < 8:
                if self.pic_num >= self.PICS_PER_SIGN:
                    self.pic_num = 0
                    self.current_sign += 1
                    self.fly_to_sign(self.current_sign)
                
                print(f"Taking pictures at sign {self.current_sign}...")
                self.fly_to_relative_pos(self.pic_num, self.sign_dict[self.current_sign])
                rospy.sleep(0.25)
                self.collect_photo(self.current_sign, self.pic_num)
                self.pic_num += 1
        print("All pictures taken...")
        return
    
    
    def fly_to_relative_pos(self, pos_num, home_pos):
        x, y, z, yaw = self.get_rel_pos(pos_num, home_pos)
        self.fly_to_pos(x, y, z, yaw)
        return
    
    def get_rel_pos(self, pos_num, home_pos):
        rel_h = self.stops[pos_num][0]
        rel_z = self.stops[pos_num][1]
        yaw_rad = math.radians(home_pos.yaw)

        new_x = home_pos.x - rel_h * math.sin(yaw_rad)
        new_y = home_pos.y + rel_h * math.cos(yaw_rad)
        new_z = max(home_pos.z + rel_z, 0.01) # make sure to stay off ground
        return  new_x, new_y, new_z, home_pos.yaw
    
    def collect_photo(self, sign_num, pic_num):
        ros_image = rospy.wait_for_message(self.camera_topic, Image, timeout=5)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            suffix = "{:04d}".format(pic_num)
            cv.imwrite(os.path.join(self.DATA_PATH, f"sign{sign_num}_{suffix}.png"), cv_image)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        return
         
if __name__ == '__main__':
    pc_node = DronePicCollectNode()
    try:
        pc_node.collect_photos()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down")