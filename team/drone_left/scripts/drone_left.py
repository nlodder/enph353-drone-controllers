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
import math

class TeamLeftDroneNode:
    def __init__(self):
        rospy.init_node('drone_left', anonymous=True)
        self.UPDATE_HZ = 30

        # STATE CONSTANTS
        self.START = 0
        self.APPROACH_STATE = 1
        self.QUERY_STATE = 2
        self.KICKOFF_STATE = 3
        self.FINISHED_STATE = 4
        self.ELEVATING = 5

        self.start_timer = 0
        self.START_CYCLES = 10
        self.started = False

        # STATE CHANGE FLAGS
        self.ready_to_approach = False
        self.ready_to_set = False
        self.ready_to_query = False
        self.ready_to_kickoff = False
        self.ready_to_look = False

        self.kickoff_timer = 0
        self.KICKOFF_CYCLES = 10
        self.KO_REC_CYCLES = 20
        self.ko_rec_timer = 0
        self.ko_rec_complete = False

        self.front_image_sub = rospy.Subscriber("camera_front/image_raw", Image, self.imageF_callback)
        self.back_image_sub = rospy.Subscriber("camera_down/image_raw", Image, self.imageD_callback)

        # CHECKING IF NN GOT READ
        self.nn_resp_received = False
        
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1) # cmd_vel for cmd_bridge node to apply as wrenches
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=10)
        self.score_sub = rospy.Subscriber("/score_tracker", String, self.score_tracker_callback)
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
        self.abs_elev_target = 0 # desired absolute elevation target for before starting line following PID
        self.altitude = None # current altitude of drone, updated from laser scan data

        self.bridge = CvBridge()
        self.window_name = f"{rospy.get_namespace()} camera feed"
        self.side_img = None
        self.down_img = None

        self.current_twist = Twist()

        # HORIZONTAL ALIGNMENT PID
        self.x_pid = PIDController(kp=50, ki=0.0, kd=0)
        self.y_pid = PIDController(kp=200, ki=0.0, kd=20)
        self.z_pid = PIDController(kp=60, ki=0.0, kd=0)
        self.error_x = 0.0
        self.error_y = 0.0
        self.error_ang_z = 0.0
        self.x_pid.previous_error=0
        self.y_pid.previous_error=0
        self.z_pid.previous_error=0
        self.x_pid.integral=0
        self.y_pid.integral=0
        self.z_pid.integral=0

        # HSV RANGE FOR SIGN COLORS
        self.LOWER_BLUE = (100, 150, 50)
        self.UPPER_BLUE = (140, 255, 255)

        # -- SETUP UNIQUE TO TEAM MEMBER (drone left vs drone right)--
        # PERSISTANT VARIABLES
        self.state = self.START
        self.current_cam = "front" # which cam feed we are currently using for image processing.
        self.current_sign = 1
        self.LAST_SIGN_NUM = 6 # all the signs this drone is responsible for inclusive

        # FOR WHICH DIRECTION DRONE NEEDS TO 'KICK OFF' WHEN LEAVING A SIGN
        SignMovements = namedtuple('SignMovements', ['x', 'y'])
        self.kickoff_movements = {
            1: SignMovements( 2, 0),
            2: SignMovements( 0.1, 0),
            3: SignMovements( 0.1, 0),
            4: SignMovements( 0.1, 0),
            5: SignMovements( 0.1, 0),
            6: SignMovements( 0.1, 0)
        }
        # PID ERROR ANALYSIS
        self.error_history = np.zeros(200)
        
        self.coord_msg.task_complete = True # BECAUSE WE DON'T HAVE THE OTHER DRONE TO SET THIS TO TRUE
        
        # WAIT FOR EVERYTHING TO INITIALIZE
        rospy.sleep(0.5)
    
    # -- CALLBACKS --
    def score_tracker_callback(self, msg):
        """
            Parses string to check if clue_type is one of 1, 2, . . . , 8.
            If it is, sets self.nn_resp_received flag to True
        """
        if self.msg_is_clue(msg.data):
            self.nn_resp_received=True
        return
    
    def alt_callback(self, msg):
        """
            Updates self.altitude to msg.data
        """
        self.altitude = msg.data
        return
    
    def coordination_callback(self, msg):
        """
            Updates self.coord_msg to value of incoming msg
        """
        self.coord_msg = msg
        return
    
    def imageF_callback(self, data):
        """
            Updates self.side_img to data if self.current_cam is "front"
        """
        if self.current_cam == "front":
            self.image_callback(data)
        return
    
    def imageD_callback(self, data):
        """
            Updates self.down_img with image converted to cv2 bgr8 format
        """
        self.down_img = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def image_callback(self, data):
        """
            Updates self.side_img with image converted to cv2 bgr8 format
        """
        self.side_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        return
    
    def msg_is_clue(self, string):
        """
            Checks if a message is a clue by parsing the image string and checking
            if the clue type field is in "1", "2", . . ., "8"
            - returns True if this case
            - returns False otherwise
        """
        team, password, clue_type, clue = string.split(",")
        print(f"{team},{password},{clue_type},{clue}")
        valid_top_words = ["1", "2", "3", "4", "5", "6", "7", "8"]
        for tw in valid_top_words:
            if clue_type == tw:
                print("VALID CLUE")
                return True
        print("INVALID CLUE")
        return False

    def is_initialized(self):
        """
            Returns True if messages have been received from node dependencies:
            - active side camera (front/back or left/right)
            - down camera
            - depth sensor

            else returns False
        """
        if self.altitude is None: return False
        if self.side_img is None: return False
        if self.down_img is None: return False
        return True

    # -- HOME BASE --
    def run(self):
        """
            Runs all processing every cycle per self.UPDATE_HZ 'home base'
            - waits until all sensor comms are established
            - publishes current state details to drone's state rostopic
            - analyzes front camera image to analyze sign readability and drone movement demands
            - updates self.state per results of analysis and previous cycle
            - updates movement demands (fwd/back, left/right, yaw, up/down) per analysis and current state
            - publishes movement demands to cmd_vel rostopic for application by subscribing node (drone_cmd_bridge)
        """
        rate = rospy.Rate(self.UPDATE_HZ)
        
        # WAIT ON SENSOR COMMUNICATION
        while not rospy.is_shutdown() and not self.is_initialized():
            state_msg = self.make_state_msg()
            self.state_pub.publish(state_msg)
            rospy.loginfo_once("Waiting for sensor data (altitude/images)...")
            rate.sleep()

        # START OFF COMP!!!!
        self.score_pub.publish(self.START_MSG)
        self.nn_resp_received = False

        # GENERAL COMP ACTIVITY
        while not rospy.is_shutdown():
            # analyze current image and update states if necessary
            self.analyze_front_img()

            # update state per results of last cycle
            self.update_state()

            # check if finished and send end msg if so
            self.endmsg_if_finished()

            # update movement demands based on image analysis and current state
            self.update_mov_demands()

            # update command bridge with movement demands
            self.abs_elev_pub.publish(self.abs_elev_target)
            self.vel_pub.publish(self.current_twist)
            state_msg = self.make_state_msg()
            self.state_pub.publish(state_msg)
            rate.sleep()
        
    def update_state(self):
        """
            Updates drone's current state based on state flags, and previous state.
            - clears all PID errors when switching from:
                - APPROACH -> QUERY
                - QUERY -> KICKOFF
                - KICKOFF -> APPROACH
            - clears state flags when executing associated state change
        """
        previous_state = self.state

        # START -> ELEVATING
        if self.state==self.START:
            self.start_timer += 1
            if self.nn_resp_received==True or self.start_timer > self.START_CYCLES:
                self.nn_resp_received = False
                self.abs_elev_target=0.3
                self.state=self.ELEVATING
        
        # ELEVATING -> KICKOFF
        elif self.state == self.ELEVATING:
            if self.altitude >= self.abs_elev_target:
                self.abs_elev_target = 0.5
                self.state = self.KICKOFF_STATE
                self.current_sign = 1
            return
        
        # APPROACH -> QUERY
        elif previous_state == self.APPROACH_STATE and self.ready_to_query:
            self.ready_to_query = False
            self.state = self.QUERY_STATE
        
        # QUERY -> KICKOFF
        elif previous_state == self.QUERY_STATE and self.ready_to_kickoff:
            if self.current_sign == 1:
                self.clear_PID_errors()
                self.state = self.ELEVATING
            else:
                self.ready_to_kickoff = False
                self.state = self.KICKOFF_STATE
        
        # KICKOFF -> APPROACH/FINISHED
        elif previous_state == self.KICKOFF_STATE:
            self.kickoff_timer += 1 
            
            # ONLY do this once the timer is actually finished
            if self.kickoff_timer > self.KICKOFF_CYCLES:
                self.ko_rec_timer = 0 
                self.ko_rec_complete = False
                self.kickoff_timer = 0
                self.current_sign += 1
                self.state = self.APPROACH_STATE
                self.clear_PID_errors()

        # CHECK IF WE'RE FINISHED.
        if self.current_sign > self.LAST_SIGN_NUM:
            self.state = self.FINISHED_STATE
            return

        return
    
    def endmsg_if_finished(self):
        """
            Publishes end msg to /score_tracker if state is FINISHED
            and the RIGHT DRONE (team partner) is also finished
        """
        if self.state == self.FINISHED_STATE:
            # check if other drone already finished, if they did, kill comp
            # LEFT DRONE is responsible for killing comp
            if self.coord_msg.task_complete == True:
                self.score_pub.publish(self.END_MSG)
        return
    
    def update_mov_demands(self):
        """
            Updates self.current_twist based on PID on errors from image analysis.
            - if state is ELEVATING, overwrites errors with constants
            - if state is START, sets all twists to 0
            - if state is KICKOFF, applies kickoff_movements
        """
        # always perform PID ...
        self.current_twist.linear.x = self.x_pid.update(self.error_x, 1.0 / self.UPDATE_HZ)
        self.current_twist.linear.y = self.y_pid.update(self.error_y, 1.0 / self.UPDATE_HZ)
        self.current_twist.angular.z = self.z_pid.update(self.error_ang_z, 1.0 / self.UPDATE_HZ)

        # smush errors if coming off kickoff
        if self.ko_rec_complete == False:
            scale = self.get_sigmoid_value(self.ko_rec_timer)
            self.current_twist.linear.x = scale * self.current_twist.linear.x
            self.current_twist.linear.y = scale * self.current_twist.linear.y
            self.current_twist.angular.z = scale * self.current_twist.angular.z
            self.ko_rec_timer += 1
            if self.ko_rec_timer > self.KO_REC_CYCLES:
                self.ko_rec_complete = True
                self.ko_rec_timer = 0
        
        # ... unless we need to jump the drone to better see the sign
        if self.state == self.KICKOFF_STATE:
            self.current_twist.linear.x = self.kickoff_movements[self.current_sign].x
            self.current_twist.linear.y = self.kickoff_movements[self.current_sign].y
            self.current_twist.angular.z = 0
            
        # here we will be at the start. Down cam will manage l/r
        elif self.state == self.ELEVATING:
            self.current_twist.linear.x = 0.05
            self.current_twist.linear.y = -0.03
            self.current_twist.angular.z = 0.0
            return
        
        if self.state == self.START:
            self.current_twist.linear.x = 0.0
            self.current_twist.linear.y = 0.0
            self.current_twist.angular.z = 0.0

        return
    
    def analyze_front_img(self):
        """
            Analyzes the image from the front camera of the drone and adjusts, based on self.state:
            - self.error_x      (self.state == all)
            - self.error_y      (self.state == ELEVATING)
            - self.error_ang_z  (self.state == all)
        """
        # make sure we've received at least one image
        if self.side_img is None:
            return
        
        bgr_img = self.side_img
        # check if the sign 
        sign_readable, area = self.sign_readable(bgr_img)    
        
        if self.state == self.APPROACH_STATE and sign_readable:
            self.ready_to_query = True

        elif self.state == self.QUERY_STATE:
            if self.nn_resp_received:
                self.ready_to_kickoff=True
                self.nn_resp_received=False

            elif sign_readable:
                self.ready_to_kickoff = False

        # X, Y, YAW ALIGNMENT ANALYSIS
        self.front_img_to_xyyaw(bgr_img)

        self.show_debug(bgr_img, self.error_ang_z, self.error_x, self.error_y, area, sign_readable)
        
        # LEFT/RIGHT ALIGNMENT ANALYSIS IF ELEVATING IS BASED ON FRONT CAM
        if self.state == self.ELEVATING:
            self.error_y = 0
            self.error_ang_z = 0
        return
    
    def front_img_to_xyyaw(self, bgr_img):
        """
            Takes image from the front camera and analyzes for error_x (fwd/bkwd)
            - if there are multiple sign signatures, takes the left-most signature
            - error_x is determined based on the height of the side of the sign
        """
        GOAL_HEIGHT_RATIO = 0.02
        GOAL_HEIGHT_RATIO_APPROACH = 0.01

        goal_ratio = GOAL_HEIGHT_RATIO
        if self.state==self.APPROACH_STATE:
            goal_ratio = GOAL_HEIGHT_RATIO_APPROACH

        # crop out the top third of the image
        img_h = bgr_img.shape[0]
        img_w = bgr_img.shape[1]

        roi_y_start = img_h // 3
        roi_y_end = img_h
        roi_bgr = bgr_img[roi_y_start:roi_y_end,  : ]

        # GET RID OF THE WHITE INSIDE THE BLUE SIGNS
        blue_contours = self.get_blue_contours(roi_bgr)
        if blue_contours:
            # take the left-most contour
            left_contour = min(blue_contours, key=lambda cnt:cv.boundingRect(cnt)[0])
            hull = cv.convexHull(left_contour)
            x, y, w, h = cv.boundingRect(hull)
            sign_height = h
            ratio = sign_height / img_h
            max_depth_error = 1-goal_ratio
            depth_error = (goal_ratio - ratio) * max_depth_error # normalize between [-1,1]
            cX = x + w//2
            self.error_x        = depth_error
            self.error_y        = (img_w//2 - cX)/(img_w//2) # normalize between [-1,1]
            self.error_ang_z    = (img_w//2 - cX)/(img_w//2) # normalize between [-1,1]

        # if no blue, set error_x to zero to prevent spinning out
        else:
            self.error_x = 0

        return

    def front_img_to_y(self, bgr_img):
        """
            Takes image from the front camera, analyzes the bottom fourth and adjusts y-error
            to centroid of the road
            - modifies self.error_y
        """
        # crop the image to the bottom fourth
        img_h = bgr_img.shape[0]
        img_w = bgr_img.shape[1]
        roi_y_start = (img_h * 5) // 8
        roi_y_end = img_h
        roi_bgr = bgr_img[roi_y_start:roi_y_end,  : ]

        M = self.get_line_moments(roi_bgr)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            img_center = img_w // 2
            cv.circle(roi_bgr, center=(cX, img_h//8), radius=20, color=(0,0,255), thickness=-1)

            # error > 0 -> shift right
            # error < 0 -> shift left
            error = img_center - cX
            self.error_y = error
        else:
            self.error_y = 0

        return

    def analyze_down_img(self):
        """
            Analyzes the image from the bottom camera of the drone and adjusts, based on self.state:
            - self.error_y      (self.state != ELEVATING)
        """
        # make sure we've received at least one image
        if self.down_img is None:
            return
        cv_image = self.down_img
        # reduce size to save compute, take full width for now
        img_h = cv_image.shape[0]
        img_w = cv_image.shape[1]
        des_offset_from_right = img_w // 4
        roi_y_start = 0 # top
        roi_y_end = img_h//2 # third of the way down
        roi_bgr = cv_image[roi_y_start:roi_y_end,  : ]

        # find center of white lines
        M = self.get_line_moments(roi_bgr)

        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            img_center = img_w // 2

            # error > 0 -> turn right
            # error < 0 -> turn left
            error = img_center - cX
            self.error_y = error

        return

    def get_line_moments(self, roi_bgr):
        """
            Gets the moments of white lines in the roi provided.
            - pass roi image in bgr8 format
            - returns moment of the white lines found with cv2.moments
            - cuts out the white from the signs using self.get_blue_contours and filling them in blue
              before masking for whites.
            - gets the line mask based on internally defined thresholding in self.get_line_mask()
            - cleans up noise in line mask with cv2.erode and cv2.dilate
            - returns moments of white lines from cv2.moments()
        """
        roi_mod = roi_bgr.copy()
        # GET RID OF THE WHITE INSIDE THE BLUE SIGNS
        blue_contours = self.get_blue_contours(roi_bgr)
        if blue_contours:
            for contour in blue_contours:
                hull= cv.convexHull(contour)
                cv.drawContours(roi_mod, [hull], -1, (255, 0, 0), thickness=-1)

        # NOW MASK FOR THE WHITES
        mask = self.get_line_mask(roi_mod)
        # clean up noise in grassa
        kernel = np.ones((11, 11), np.uint8)
        mask = cv.erode(mask, kernel, iterations=1)
        mask = cv.dilate(mask, kernel, iterations=2)

        # find center of white lines
        M = cv.moments(mask)
        return M

    def get_line_mask(self, roi_bgr):
        """
            Takes bgr image and returns a mask of the white pixels based on an internally
            defined color range.
        """
        roi_hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)

        # low saturation because white is low saturation whereas grass speckles may be high
        lower_white = np.array([0,0,200])
        upper_white = np.array([180,40,255])

        mask = cv.inRange(roi_hsv, lower_white, upper_white)
        return mask

    def sign_readable(self, cv_image):
        """
            Masks image for the blue color of the sign.
            If sign is fully in frame and large enough, return True to indicate that sign may be readable by our neural network.
            - conditions for readability:
                - threshold_upper > contour area > threshold_lower (to ensure sign is large enough in frame but not just a corner)
                - centroid of contour associated with sign border is near center of image
                - modifies the x_error and y_error for PID centering if sign border is detected, to try to center the drone over the sign for better reads.
        """
        THRSHOLD_LOWER = 3000
        THRSHOLD_UPPER = 50000
        sign_readable = False
        area=0

        try:
            roi_bgr = cv_image
            contours = self.get_blue_contours(roi_bgr)
            if len(contours) > 0:
                # take convex hull of contours to overcome interior countour paths
                largest_contour = max(contours, key=cv.contourArea)
                hull = cv.convexHull(largest_contour) # smoothing edges
                cv.drawContours(cv_image, [hull], -1, (0,0,255), 2)

                area = cv.contourArea(hull)
                M = cv.moments(hull)
                if M["m00"] == 0: return False
                # horizontal and vertical coordinates of centroid.
                cXY = int(M["m10"] / M["m00"])
                cZ = int(M["m01"] / M["m00"])

                cv.circle(cv_image, (cXY, cZ), 7, (0,0,255), -1)

                # check if contour area is within thresholds and centroid is near center of image
                # don't worryabout centering in height, only in xy
                if THRSHOLD_LOWER < area < THRSHOLD_UPPER:
                    sign_readable = True
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        return sign_readable, area
    
    def get_blue_contours(self, roi_bgr):
        """
            Returns the contours of the blue sign candidates found in an roi
            - pass in roi in bgr8 format
            - masks based on self.LOWER_BLUE, self.UPPER_BLUE
        """
        hsv_image = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)
        # filter for a range around common 'blue'
        mask = cv.inRange(hsv_image, self.LOWER_BLUE, self.UPPER_BLUE)

        # this should close any 'broken' edges...
        kernel = np.ones((20, 20), np.uint8)
        solid_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        contours = cv.findContours(solid_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        return contours

    def make_state_msg(self):
        """
            Creates a string state message including:
            - current state
            - x velocity
            - y velocity
            - kickoff cycle
            - start cycle
        """
        # Map integer states to readable strings
        state_names = {
            self.START: "starting",
            self.ELEVATING: "elevating",
            self.APPROACH_STATE: "approaching",
            self.QUERY_STATE:    "querying",
            self.KICKOFF_STATE:  "kick off",
            self.FINISHED_STATE: "finished"
        }
        
        current_state_str = state_names.get(self.state, "unknown")
        
        # Create a formatted multi-line string with fixed-width columns
        msg = (
            f"{'current state:':<15} {current_state_str}"
            f"{' || current sign:':<15} {self.current_sign}"
            f"{' || velocity x:':<15} {self.current_twist.linear.x:>8.3f}"
            f"{' || velocity y:':<15} {self.current_twist.linear.y:>8.3f}"
            f"{' || kickoff cycle:':<15} {self.kickoff_timer:>8.1f}"
            f"{' || start cycle:':<15} {self.start_timer:>8.1f}"
        )
        return msg 

    def draw_error_plot(self, frame):
        """
            Draws a scrolling error plot across the top of the frame.
        """
        plot_h = 80  # Height of the plot area
        plot_w = 200 # Width of the plot (matching history buffer)
        img_w = frame.shape[1]
        
        # update history (shift left and add current error)
        # normalizing error so it fits in the plot_h (adjust 100 based on typical max error)
        norm_error = np.clip(self.error_y, -100, 100) 
        self.error_history = np.roll(self.error_history, -1)
        self.error_history[-1] = norm_error

        # create a black background for the plot
        # placing it at the top-right
        cv.rectangle(frame, (img_w - plot_w - 10, 10), (img_w - 10, 10 + plot_h), (0, 0, 0), -1)
        cv.rectangle(frame, (img_w - plot_w - 10, 10), (img_w - 10, 10 + plot_h), (100, 100, 100), 1)
        
        # draw Center Reference Line
        ref_y = 10 + (plot_h // 2)
        cv.line(frame, (img_w - plot_w - 10, ref_y), (img_w - 10, ref_y), (50, 50, 50), 1)

        # draw the Plot Line
        for i in range(1, len(self.error_history)):
            # Map history values to pixel coordinates
            pt1_x = (img_w - plot_w - 10) + (i - 1)
            pt2_x = (img_w - plot_w - 10) + i
            
            # Scale error: plot_h//2 is the middle, 0.4 is a scaling factor
            pt1_y = ref_y - int(self.error_history[i-1] * 0.4)
            pt2_y = ref_y - int(self.error_history[i] * 0.4)
            
            # Draw the line segment
            cv.line(frame, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 1)

        # add Labels
        cv.putText(frame, "Y-Error Hist", (img_w - plot_w - 10, 25), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) 
    
    def show_debug(self, bgr_img, error_ang_z, error_x, error_y, area, sign_readable):
        # Create the text string
        textz = f"Error Ang Z: {error_ang_z:.2f}"
        texty = f"Error Lin Y: {error_y:.2f}"
        textx = f"Error Lin X: {error_x:.2f}"
        textA = f"Area: {area:.2f} || Readable: {sign_readable}"
        
        cv.putText(img=bgr_img, text=textx, org=(20, 40), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.7, color=(0, 0, 255),thickness=2)
        cv.putText(img=bgr_img, text=texty, org=(20, 60), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.7, color=(0, 0, 255),thickness=2)
        cv.putText(img=bgr_img, text=textz, org=(20, 80), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.7, color=(0, 0, 255),thickness=2)
        cv.putText(img=bgr_img, text=textA, org=(20, 100), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.7, color=(0, 0, 255),thickness=2)
        cv.imshow("Front Image", bgr_img)
        cv.waitKey(1)
        return
    
    def clear_PID_errors(self):
        """
            Clears integral and derivative errors in both x and y.
            Use this when switching to a new camera view or PID setpoint
        """
        self.x_pid.integral = 0
        self.z_pid.integral = 0
        self.y_pid.integral = 0
        self.x_pid.previous_error = 0
        self.z_pid.previous_error = 0
        self.y_pid.previous_error = 0 
    
    def get_sigmoid_value(self, current_cycle):
        """
            Returns the value of the sigmoid function at the current timestep
            where the sigmoid goes from 0 to 1 over self.KO_REC_CYCLES
        """
        # k=10 ensures that at cycle 0, output is ~0.006 
        # and at max cycle, output is ~0.993
        k = 10 
        d=0.5
        
        # normalize current cycle to 0.0 - 1.0
        normalized_x = float(current_cycle) / float(self.KO_REC_CYCLES)
        
        # Shift and apply steepness
        z = k * (normalized_x - d)
        
        # Calculate Sigmoid
        return 1 / (1 + math.exp(-z))

class PIDController:
    """Simple PID controller"""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
    
    def update(self, error, dt):
        """
            Simple PID controller.
            - error: difference between map centroid and drone frame centre
            - dt: time step in seconds
            Returns the control output velocity to apply.
        """
        leak_coeff = 0.9
        self.integral += leak_coeff * error * dt
        self.integral = max(-1.0, min(1.0, self.integral)) # clamp integral to prevent runaway
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