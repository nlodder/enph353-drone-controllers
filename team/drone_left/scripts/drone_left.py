#!/usr/bin/env python3
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64, String
from drone_msgs.msg import DroneMessage
import numpy as np
from collections import namedtuple

class TeamLeftDroneNode:
    def __init__(self):
        rospy.init_node('drone_left', anonymous=True)
        self.UPDATE_HZ = 30

        # STATE CONSTANTS
        self.LOOKING_STATE = 0
        self.APPROACH_STATE = 1
        self.SETTING_STATE = 2
        self.QUERY_STATE = 3
        self.KICKOFF_STATE = 4
        self.FINISHED_STATE = 5
        self.ELEVATING = 6

        # STATE CHANGE FLAGS
        self.ready_to_approach = False
        self.ready_to_set = False
        self.ready_to_query = False
        self.ready_to_kickoff = False
        self.ready_to_look = False

        self.kickoff_timer = 0
        self.KICKOFF_CYCLES = 30
        
        # Subscribe to the camera topic defined in Gazebo plugin
        self.right_image_sub = rospy.Subscriber("camera_right/image_raw", Image, self.imageR_callback)
        self.left_image_sub = rospy.Subscriber("camera_left/image_raw", Image, self.imageL_callback)
        self.front_image_sub = rospy.Subscriber("camera_front/image_raw", Image, self.imageF_callback)
        self.back_image_sub = rospy.Subscriber("camera_back/image_raw", Image, self.imageB_callback)

        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1) # cmd_vel for cmd_bridge node to apply as wrenches
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=10)
        self.state_pub = rospy.Publisher("state", String, queue_size = 10)

        # COORDINATION WITH OVERSEER
        self.coord_pub = rospy.Publisher("/drone_coordination", DroneMessage, queue_size=10)
        self.coord_sub = rospy.Subscriber("/drone_coordination", DroneMessage, self.coordination_callback)
        self.coord_msg = DroneMessage()
        
        # COMPETTITION START/END MESSAGES
        self.TEAM_NAME = "CYRUS"
        self.TEAM_PASS = "1111"
        self.START_MSG = f"{self.TEAM_NAME},{self.TEAM_PASS},0,aaaa"
        self.END_MSG = f"{self.TEAM_NAME},{self.TEAM_PASS},-1,aaaa"

        self.abs_elev_pub = rospy.Publisher("abs_z_target", Float64, queue_size=1) # publish absolute elevation target for cmd bridge to maintain for oversight
        self.alt_sub = rospy.Subscriber("altitude", Float64, self.alt_callback)
        self.abs_elev_target = 0.5 # desired absolute elevation target for oversight, adjust as needed
        self.altitude = None # current altitude of drone, updated from laser scan data

        self.bridge = CvBridge()
        self.window_name = f"{rospy.get_namespace()} camera feed"
        self.current_image = None

        self.current_twist = Twist()

        # HORIZONTAL ALIGNMENT PID
        self.x_pid = PIDController(kp=0.004, ki=0.0, kd=0.008)
        self.y_pid = PIDController(kp=0.004, ki=0.0, kd=0.008)
        self.error_x = 0.0
        self.error_y = 0.0

        # HSV RANGE FOR SIGN COLORS
        self.LOWER_BLUE = (100, 150, 50)
        self.UPPER_BLUE = (140, 255, 255)

        # -- SETUP UNIQUE TO TEAM MEMBER --
        # PERSISTANT VARIABLES
        self.state = self.ELEVATING
        self.current_cam = "right" # which cam feed we are currently using for image processing.
        self.current_sign = 1
        self.LAST_SIGN_NUM = 4 # all the signs this drone is responsible for inclusive

        # FOR WHICH DIRECTION DRONE NEEDS TO 'JUMP' WHEN ARRIVING AT SIGN TO READ IT
        SignMovements = namedtuple('SignMovements', ['x', 'y'])
        self.setting_movements = {
            1: SignMovements(  0,  0),
            2: SignMovements(  0,  0),
            3: SignMovements(  0,  0),
            4: SignMovements( -1, -1)
        }

        # FOR WHICH DIRECTION DRONE NEEDS TO 'KICK OFF' WHEN LEAVING A SIGN
        self.kickoff_movements = {
            1: SignMovements( -0.2, -1),
            2: SignMovements(   -1, -1),
            3: SignMovements(   -1,  1),
            4: SignMovements(    0,  0)
        }

        # FOR TRACKING WHICH CAMERA THE DRONE SHOULD BE USING
        SignInfo = namedtuple('SignInfo', ['approach_cam', 'setting_cam', 'clue_type'])
        self.sign_dict = {
            1: SignInfo("right", "right", "SIZE"),
            2: SignInfo("right", "right", "VICTIM"),
            3: SignInfo("back", "back", "CRIME"),
            4: SignInfo("back", "left", "TIME")
        }


        rospy.sleep(0.5)
        # START OFF COMP!!!!
        self.score_pub.publish(self.START_MSG)
    
    # -- CALLBACKS --
    def alt_callback(self, msg):
        self.altitude = msg.data
    
    def coordination_callback(self, msg):
        self.coord_msg = msg
        return

    # CALLBACKS FOR FOUR CAMERAS - only process image if from cam of interest
    def imageR_callback(self, data):
        if self.current_cam == "right":
            self.image_callback(data)
        return
    def imageL_callback(self, data):
        if self.current_cam == "left":
            self.image_callback(data)
        return
    def imageF_callback(self, data):
        if self.current_cam == "front":
            self.image_callback(data)
        return
    def imageB_callback(self, data):
        if self.current_cam == "back":
            self.image_callback(data)
        return

    def image_callback(self, data):
        """Updates self.current_image with image converted to cv2 bgr8 format"""
        self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        return
    
    def is_initialized(self):
        """
            Returns true if messages have been recieved from node dependencies:
            - cameras
            - depth sensor
        """
        if self.altitude is None: return False
        if self.current_image is None: return False
        return True

    def run(self):
        """
        Runs state machine
        - state = looking       -> search for sign signature in cam associated with self.current_sign
        - state = approaching   -> PID on object scale and centering
        - state = setting       -> executing shift if necessary and centering
        - state = querying      -> sending images to nn and waiting for responses. publish to score_tracker if good. increment sign & state
        - state = kickoff       -> kicking off from read sign so as to not re-read it
        - state = finished      -> publish to other drone to let them know we're done.
                                   if they are done, publish to score tracker
        """
        rate = rospy.Rate(self.UPDATE_HZ)
        
        # wait until sensor comms established
        while not rospy.is_shutdown() and not self.is_initialized():
            rospy.loginfo_once("Waiting for sensor data (altitude/images)...")
            rate.sleep()
        
        # get us to target elevation for main phase
        self.abs_elev_pub.publish(self.abs_elev_target)
        
        while not rospy.is_shutdown():
            # analyze current image and update states if necessary
            self.analyze_image()

            # update movement demands based on image analysis and current state
            self.update_mov_demands()

            # update state per results of last cycle
            self.update_state()
            
            # update command bridge with movement demands
            self.vel_pub.publish(self.current_twist)
            state_msg = self.make_state_msg()
            self.state_pub.publish(state_msg)
            rate.sleep()
        
    def update_state(self):
        """
            Updates the state of the drone based on self.current_sign value and
            state change flags.
            - increments self.current_sign if self.ready_to_look == True
            - switches self.current_cam when switching in to LOOKING or SETTING states
            per self.sign_dict[self.current_sign]
        """
        previous_state = self.state

        # INITIAL ELEVATION BEFORE ANYTHING ELSE
        if self.state == self.ELEVATING:
            if self.altitude >= self.abs_elev_target:
                self.state = self.QUERY_STATE
            return

        elif self.current_sign > self.LAST_SIGN_NUM:
            self.state = self.FINISHED_STATE
            return

        elif previous_state == self.LOOKING_STATE and self.ready_to_approach:
            self.ready_to_approach = False
            self.state = self.APPROACH_STATE

        elif previous_state == self.APPROACH_STATE and self.ready_to_set:
            self.ready_to_set = False
            self.state = self.SETTING_STATE
            self.clear_PID_errors()
            self.update_current_cam()
        
        elif previous_state == self.SETTING_STATE and self.ready_to_query:
            self.ready_to_query = False
            self.state = self.QUERY_STATE
        
        elif previous_state == self.QUERY_STATE and self.ready_to_kickoff:
            self.ready_to_kickoff = False
            self.state = self.KICKOFF_STATE
        
        elif previous_state == self.KICKOFF_STATE:
            self.kickoff_timer += 1 
            
            # ONLY do this once the timer is actually finished
            if self.kickoff_timer > self.KICKOFF_CYCLES: 
                self.kickoff_timer = 0
                self.current_sign += 1
                
                # Check if we are done after incrementing
                if self.current_sign > self.LAST_SIGN_NUM:
                    self.state = self.FINISHED_STATE
                else:
                    self.state = self.LOOKING_STATE
                    self.clear_PID_errors()
                    self.update_current_cam()
        
        if self.state == self.FINISHED_STATE:
            # check if other drone already finished, if they did, kill comp
            # LEFT DRONE is responsible for killing comp
            if self.coord_msg.task_complete == True:
                self.score_pub.publish(self.END_MSG)

        return

    
    def update_current_cam(self):
        """
            Updates self.current cam to be cam associated with self.current_sign
            and self.current_state (either SETTING or LOOKING).
            If in neither of these states, this function leaves self.current_cam
            as it was.
        """
        if self.state == self.SETTING_STATE:
            self.current_cam = self.sign_dict[self.current_sign].setting_cam
        elif self.state == self.LOOKING_STATE:
            self.current_cam = self.sign_dict[self.current_sign].approach_cam
        return
    
    def clear_PID_errors(self):
        """
            Clears integral and derivative errors in both x and y.
            Use this when switching to a new camera view or PID setpoint
        """
        self.x_pid.integral = 0
        self.y_pid.integral = 0
        self.x_pid.previous_error = 0
        self.y_pid.previous_error = 0
    
    def update_mov_demands(self):
        """
            Checks if map centroid - frame centroid error < threshold
            - if map centroid - frame centroid error < threshold for 1s, then switch self.state = "commanding"
            - always performs PID control to try to prevent drifting error due to wind
        """
        # always perform PID ...
        self.current_twist.linear.x = self.x_pid.update(self.error_x, 1.0 / self.UPDATE_HZ)
        self.current_twist.linear.y = self.y_pid.update(self.error_y, 1.0 / self.UPDATE_HZ)
        
        # ... unless we need to jump the drone to better see the sign
        if self.state == self.SETTING_STATE:
            self.current_twist.linear.x = self.setting_movements[self.current_sign].x
            self.current_twist.linear.y = self.setting_movements[self.current_sign].y
        elif self.state == self.KICKOFF_STATE:
            self.current_twist.linear.x = self.kickoff_movements[self.current_sign].x
            self.current_twist.linear.y = self.kickoff_movements[self.current_sign].y
        elif self.state == self.ELEVATING:
            # self.current_twist.linear.x = 0
            # self.current_twist.linear.y = 0
            return

        return
    
    def analyze_image(self):
        # make sure we've recieved at least one image
        if self.current_image is None:
            return
        cv_image = self.current_image
        sign_readable = self.sign_readable(cv_image)    
        
        if self.state == self.LOOKING_STATE and sign_readable:
            self.ready_to_approach = True

        elif self.state == self.APPROACH_STATE and sign_readable:
            self.ready_to_set = True

        elif self.state == self.SETTING_STATE and sign_readable:
            self.ready_to_query = True

        elif self.state == self.QUERY_STATE and sign_readable:
            # TODO: MAKE THIS COMMUNICATION WITH NN ROSTOPIC
            read_good, clue_prediction = self.read_sign(cv_image)
            if read_good:
                self.score_pub.publish(f"{self.TEAM_NAME},{self.TEAM_PASS},{self.current_sign},{clue_prediction}")
                self.ready_to_kickoff = True
            else:
                self.ready_to_kickoff = False
        
        return
    
    def read_sign(self, cv_image):
        """
            Uses a neural network to read the sign in the image and return a confidence in the read and the predicted clue.
            For now, this is just a placeholder that returns random values. You can replace this with your actual neural network inference code.
        """
        return True, "THIS"

    def sign_readable(self, cv_image):
        """
        Masks image for the blue color of the sign. If the sign is fully in frame and also large enough,
        return true to indicate that sign may be readable by our neural network.
        - conditions for readability:
            - threshold_upper > contour area > threshold_lower (to ensure sign is large enough in frame but not just a corner)
            - centroid of contour associated with sign border is near center of image
            - modifies the x_error and y_error for PID centering if sign border is detected, to try to center the drone over the sign for better reads.
        """
        THRSHOLD_LOWER = 1000
        THRSHOLD_UPPER = 50000
        GOAL_AREA_RATIO = 0.05 
        GOAL_AREA_RATIO_APPROACH = 0.005
        CENTER_TOLERANCE = 50
        sign_readable = False
        try:
            hsv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
            # filter for a range around common 'blue'
            mask = cv.inRange(hsv_image, self.LOWER_BLUE, self.UPPER_BLUE)

            # this should close any 'broken' edges...
            kernel = np.ones((20, 20), np.uint8)
            solid_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            contours = cv.findContours(solid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            
            if len(contours) > 0:
                # take convex hull of contours to overcome countour paths into centre of map (walls tied to roads etc.)
                largest_contour = max(contours, key=cv.contourArea)
                hull = cv.convexHull(largest_contour) # smoothing edges
                cv.drawContours(cv_image, [hull], -1, (0,0,255), 2)

                area = cv.contourArea(hull)
                M = cv.moments(hull)
                if M["m00"] == 0: return False
                # horizontal and vertical coordinates of centroid.
                cXY = int(M["m10"] / M["m00"])
                cZ = int(M["m01"] / M["m00"])
                image_width = cv_image.shape[1]
                image_height = cv_image.shape[0]

                # calculate linearized area ratio
                # use sqrt to get into pixel-width units                
                area_ratio = area / (image_width * image_height)
                current_ratio_sqrt = np.sqrt(area_ratio)
                goal_ratio_sqrt = np.sqrt(GOAL_AREA_RATIO)
                if self.state == self.LOOKING_STATE or self.state==self.APPROACH_STATE:
                    goal_ratio_sqrt = np.sqrt(GOAL_AREA_RATIO_APPROACH)                    

                # pass linearize values to modify the xy errors
                self.modify_errors(cXY, image_width, image_height, current_ratio_sqrt, goal_ratio_sqrt)

                cv.circle(cv_image, (cXY, cZ), 7, (0,0,255), -1)

                # check if contour area is within thresholds and centroid is near center of image
                # don't worryabout centering in height, only in xy
                if THRSHOLD_LOWER < area < THRSHOLD_UPPER and abs(cXY - image_width//2) < CENTER_TOLERANCE:
                    sign_readable = True
                
            cv.imshow(self.window_name, cv_image)
            cv.waitKey(1)
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        return sign_readable

    def modify_errors(self, cXY, img_w, img_h, current_ratio_sqrt, goal_ratio_sqrt):
        """
            Modifies the self.error_x and self.error_y based on the position of the sign centroid and current camera in use
            - cXY is the horizontal coordinate of the centroid of the detected sign border contour
            - updates self.error_y or self.error_x to try to center the drone over the sign horizontally
        """
        # positive if target left, negative if target right
        centroid_error = (img_w//2) - cXY
        # positive if target far, negative if target close
        depth_error = (goal_ratio_sqrt - current_ratio_sqrt) * img_w
        
        if self.current_cam == "right":
            # camera looks at +Y, centroid error moves drone on X, depth error moves drone on Y
            self.error_x = -centroid_error
            self.error_y = -depth_error

        elif self.current_cam == "left":
            # camera looks at -Y, centroid error moves drone on X, depth error moves drone on Y
            self.error_x = centroid_error
            self.error_y = depth_error

        elif self.current_cam == "front":
            # camera looks at -X, centroid error moves drone on Y, depth error moves drone on X
            self.error_x = depth_error
            self.error_y = centroid_error
        
        elif self.current_cam == "back":
            # camera looks at +X, centroid error moves drone on Y, depth error moves drone on X
            self.error_x = -depth_error
            self.error_y = -centroid_error
    
        return   

    def make_state_msg(self):
        # Map integer states to readable strings
        state_names = {
            self.ELEVATING: "elevating",
            self.LOOKING_STATE:  "looking",
            self.APPROACH_STATE: "approaching",
            self.SETTING_STATE:  "setting",
            self.QUERY_STATE:    "querying",
            self.KICKOFF_STATE:  "kick off",
            self.FINISHED_STATE: "finished"
        }
        
        current_state_str = state_names.get(self.state, "unknown")
        
        # Create a formatted multi-line string with fixed-width columns
        msg = (
            f"{'DRONE STATUS: ':^30}"
            f"{' || current state:':<15} {current_state_str}"
            f"{' || current sign:':<15} {self.current_sign}"
            f"{' || velocity x:':<15} {self.current_twist.linear.x:>8.3f}"
            f"{' || velocity y:':<15} {self.current_twist.linear.y:>8.3f}"
            f"{' || kickoff cycle:':<15} {self.kickoff_timer:>8.3f}"
        )
        return msg   

class PIDController:
    """Simple PID controller for stabilizing in plane."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
    
    def update(self, error, dt):
        """
        Simple PID controller to stabilize the drone's position when it has owner ship.
         - error: difference between map centroid and drone frame centre
         - dt: time step in seconds
         Returns the control output velocity to apply.
        """
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

def main():
    try:
        bridge = TeamLeftDroneNode()
        bridge.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()