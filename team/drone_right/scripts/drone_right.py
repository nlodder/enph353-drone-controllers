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

class TeamRightDroneNode:
    def __init__(self):
        rospy.init_node('drone_right', anonymous=True)
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

        # -- SETUP UNIQUE TO TEAM MEMBER
        # PERSISTANT VARIABLES
        self.state = self.ELEVATING
        self.current_cam = "back" # which cam feed we are currently using for image processing.
        self.current_sign = 4
        self.LAST_SIGN_NUM = 8

        # FOR WHICH DIRECTION DRONE NEEDS TO 'JUMP' WHEN ARRIVING AT SIGN TO READ IT
        SignMovements = namedtuple('SignMovements', ['x', 'y'])
        self.setting_movements = {
            5: SignMovements( -1,  1),
            6: SignMovements(  0,  0),
            7: SignMovements( -1, -1),
            8: SignMovements(  0,  0)
        }

        # FOR WHICH DIRECTION DRONE NEEDS TO 'KICK OFF' WHEN LEAVING A SIGN
        self.kickoff_movements = {
            4: SignMovements( -0.1,  0),
            5: SignMovements( -1,  0),
            6: SignMovements(  0,  0),
            7: SignMovements(  0,  1),
            8: SignMovements(  0,  0)
        }

        # FOR TRACKING WHICH CAMERA THE DRONE SHOULD BE USING
        SignInfo = namedtuple('SignInfo', ['approach_cam', 'setting_cam', 'clue_type'])
        self.sign_dict = {
            4: SignInfo("back", "back", "NIL"),
            5: SignInfo("back", "right", "PLACE"),
            6: SignInfo("back", "back", "MOTIVE"),
            7: SignInfo("right", "front", "WEAPON"),
            8: SignInfo("front", "front", "BANDIT")
        }

        rospy.sleep(0.5)
    
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
                self.state = self.KICKOFF_STATE
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
            # tell LEFT drone that RIGHT drone is finished
            self.coord_msg.task_complete = True
            self.coord_pub.publish(self.coord_msg)

        return

    
    
    def update_current_cam(self):
        """
            Updates self.current cam to be cam associated with self.current_sign
            and self.state (either SETTING or LOOKING).
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
            self.current_twist.linear.x = -0.1

        return

    def analyze_image(self):
        # make sure we've recieved at least one image
        if self.current_image is None:
            return
        cv_image = self.current_image
        sign_readable = self.sign_readable(cv_image)
        
        if self.state == self.LOOKING_STATE:
            if self.sign_located(cv_image):
                self.ready_to_approach = True
            else:
                return # so we don't waste time on remaining conditionals

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
    
    def sign_located(self, cv_image):
        """
        - Returns true if a sign was located in the image
        - modifies self.error_x and self.error_y if sign located through calling
          self.modify_errors
        - looks first in lower 2/3 before analyzing full img to keep it light
        """
        # chop image down to see if we can find a sign in the lower 2/3 and save compute
        image_width = cv_image.shape[1]
        image_height = cv_image.shape[0]

        # try smaller img size
        roi = cv_image[ int(image_height*1/3) : , :]
        # process smaller image
        try:
            sorted_contours = None

            # ROI attempt
            roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            contours = self.blue_contours(roi_hsv)

            if not contours:
                hsv_img = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
                contours = self.blue_contours(hsv_img)

            if not contours:
                return False
            
            # grab the top 4 contours based on area
            sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
            top_x = min(len(sorted_contours), 4)
            top_contours = sorted_contours[ : top_x]

            # find the closest sign based on how close to bottom of img its bottom edge is
            closest_sign = None
            max_y = -1
            for cnt in top_contours:
                x, y, w, h = cv.boundingRect(cnt)
                bottom_y = y + h # The bottom edge of the sign

                if bottom_y > max_y:
                    max_y = bottom_y
                    closest_sign = (x,y,w,h)
            
            bx, by, bw, bh = closest_sign
            cXY = bx + bw//2
            # now adjust errors for PID based on the closest sign
            self.modify_errors(cXY, image_width, image_height, bh, 0.05)
        
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        return True
    
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

        GOAL_HEIGHT_RATIO = 0.05 
        GOAL_HEIGHT_RATIO_APPROACH = 0.005

        CENTER_TOLERANCE = 50
        is_readable = False

        try:
            contours = self.blue_contours(cv_image)
            
            if len(contours) > 0:
                # take convex hull of contours to overcome countour paths into centre of map (walls tied to roads etc.)
                largest_contour = max(contours, key=cv.contourArea)
                x, y, w, h = cv.boundingRect(largest_contour)
                # hull = cv.convexHull(largest_contour) # smoothing edges
                cv.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.drawContours(cv_image, contours, -1, (0,255,0), 2)

                cXY = x + w//2
                cZ = y + h//2

                image_width = cv_image.shape[1]
                image_height = cv_image.shape[0]

                goal_ratio = GOAL_HEIGHT_RATIO
            
                if self.state == self.LOOKING_STATE or self.state==self.APPROACH_STATE:
                    goal_ratio = GOAL_HEIGHT_RATIO_APPROACH                  

                # pass linearize values to modify the xy errors
                self.modify_errors(cXY, image_width, image_height, h, GOAL_HEIGHT_RATIO)

                cv.circle(cv_image, (cXY, cZ), 7, (0,255,0), -1)

                area = cv.contourArea(largest_contour)
                is_centered = abs(cXY - image_width//2) < CENTER_TOLERANCE
                is_sized = THRSHOLD_LOWER < area < THRSHOLD_UPPER

                if is_centered and is_sized:
                    is_readable = True
                
            cv.imshow(self.window_name, cv_image)
            cv.waitKey(1)
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            
        return is_readable

    def modify_errors(self, cXY, img_w, img_h, sign_height, goal_height_ratio=0.2):
        """
            Modifies the self.error_x and self.error_y based on the position of the sign centroid and current camera in use
            - cXY is the horizontal coordinate of the centroid of the detected sign border contour
            - updates self.error_y or self.error_x to try to center the drone over the sign horizontally
        """
        # positive if target left, negative if target right
        centroid_error = (img_w//2) - cXY
        # positive if target far, negative if target close
        current_ratio = sign_height / img_h
        depth_error = (goal_height_ratio - current_ratio) * img_w
        
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

    def blue_contours(self, bgr_image):
        """
        Returns a contours of the the blues in range [self.LOWER_BLUE, self.UPPER_BLUE]
        - pass as argument a bgr8 image
        """
        try:
            hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
            # filter for a range around common 'blue'
            mask = cv.inRange(hsv_image, self.LOWER_BLUE, self.UPPER_BLUE)
            # this should close any 'broken' edges...
            kernel = np.ones((20, 20), np.uint8)
            solid_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            contours = cv.findContours(solid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        return contours if contours else []
    
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
        bridge = TeamRightDroneNode()
        bridge.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()