#!/usr/bin/env python3
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64, String
from drone_msgs.msg import DroneMessage
import numpy as np

class WorkerDroneNode:
    def __init__(self):
        rospy.init_node('worker', anonymous=True)
        self.UPDATE_HZ = 30
        
        # Subscribe to the camera topic defined in Gazebo plugin
        self.right_image_sub = rospy.Subscriber("camera_right/image_raw", Image, self.imageR_callback)
        self.left_image_sub = rospy.Subscriber("camera_left/image_raw", Image, self.imageL_callback)
        self.front_image_sub = rospy.Subscriber("camera_front/image_raw", Image, self.imageF_callback)
        self.back_image_sub = rospy.Subscriber("camera_back/image_raw", Image, self.imageB_callback)

        self.vel_pub = rospy.Publisher("worker/cmd_vel", Twist, queue_size=1) # cmd_vel for cmd_bridge node to apply as wrenches
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=10)

        # COORDINATION WITH OVERSEER
        self.coord_pub = rospy.Publisher("/drone_coordination", DroneMessage, queue_size=10)
        self.coord_sub = rospy.Subscriber("/drone_coordination", DroneMessage, self.coordination_callback)
        self.coord_msg = DroneMessage()
        self.OVERSEER = 0
        self.WORKER = 1
        self.ready_to_request = False # whether we have good view of sign and want to request control from overseer
        self.clue_published = False # whether we have published the clue for the current sign
        self.current_cam = "right" # which cam feed we are currently using for image processing.
        
        # COMPETTITION START/END MESSAGES
        self.TEAM_NAME = "CYRUS"
        self.TEAM_PASS = "1111"
        self.START_MSG = f"{self.TEAM_NAME},{self.TEAM_PASS},0,aaaa"
        self.END_MSG = f"{self.TEAM_NAME},{self.TEAM_PASS},-1,aaaa"

        self.abs_elev_pub = rospy.Publisher("abs_z_target", Float64, queue_size=1) # publish absolute elevation target for cmd bridge to maintain for oversight
        self.alt_sub = rospy.Subscriber("altitude", Float64, self.alt_callback)
        self.abs_elev_target = 0.5 # desired absolute elevation target for oversight, adjust as needed
        self.altitude = 0.01 # current altitude of drone, updated from laser scan data

        self.bridge = CvBridge()
        self.window_name = f"Worker camera feed: {rospy.get_namespace()}"

        self.current_twist = Twist()
        self.state = "centering" # working, centering, commanding

        # HORIZONTAL ALIGNMENT PID
        self.x_pid = PIDController(kp=0.005, ki=0.0, kd=0.001)
        self.y_pid = PIDController(kp=0.005, ki=0.0, kd=0.001)
        self.error_x = 0.0
        self.error_y = 0.0

        rospy.sleep(0.5)
    
    def alt_callback(self, msg):
        self.altitude = msg.data
    
    def coordination_callback(self, msg):
        self.coord_msg = msg
        # TODO: determine which camera is the current_cam
        return

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
        """
            Processes incoming camera images for centering the drone over the map and will eventually be used for commanding other robot to follow a path.
            state = "init_rise" -> just rise up, ignore images
            state = "centering" -> process images to center over map, then switch to commanding
            - Uses Canny edge detection and contour finding to identify the largest contour in the image (assumed to be the map)
            - Computes the centroid of the largest contour and calculates the error from the center of the image
            - Updates self.error_x and self.error_y for use in PID control for centering
            - Displays the processed image with detected contours and centroids for debugging purposes.
            state = "commanding" -> process images to maintain centering, and publish commands for other robot to follow path
        """
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        if self.state == "working":
            # just check if we can identify a sign well enough to request control from overseer for PID fine tuning
            # if we can, set self.ready_to_request=True
            self.ready_to_request = self.sign_readable(cv_image)
        elif self.state == "owning":
            # try to read sign. If confidence in read is high enough, publish to score_tracker topic and
            # increment sign number in /drone_coordination msg to inform overseer of score and switch back to working state
            sign_confidence, clue_prediction = self.read_sign(cv_image)
            if sign_confidence > 0.8:
                self.score_pub.publish(f"{self.TEAM_NAME},{self.TEAM_PASS},{self.coord_msg.active_sign},{clue_prediction}")
                self.coord_pub.publish(self.coord_msg)
                self.clue_published = True
        return
    

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
        CENTER_TOLERANCE = 50
        sign_readable = False
        try:
            hsv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
            # filter for a range around common 'blue'
            lower_blue = (100, 150, 50)
            upper_blue = (140, 255, 255)
            mask = cv.inRange(hsv_image, lower_blue, upper_blue)

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

                # pass linearize values to modify the xy errors
                self.modify_errors(cXY, cZ, image_width, image_height, current_ratio_sqrt, goal_ratio_sqrt)

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

    def modify_errors(self, cXY, cZ, img_w, img_h, current_ratio_sqrt, goal_ratio_sqrt):
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

    def run(self):
        """
            Runs state machine for overseer
            - state = read1      -> read first sign
            - state = centering  -> altitude PID and xy PID, next state if centered for 1s
            - state = commanding -> 
        """
        rate = rospy.Rate(self.UPDATE_HZ)
        # get us to target elevation for main phase
        self.abs_elev_pub.publish(self.abs_elev_target)
        
        while not rospy.is_shutdown():
            # sync state with incoming coordination messages
            self.update_state()

            if self.state == "working":
                # if state is working under overseer, check if we have good view of sign
                # if we have good view of sign, request control from overseer
                # by publishing to /drone_coordination with worker_ready=True
                continue

            elif self.state == "owning":
                self.execute_centering()

            self.vel_pub.publish(self.current_twist)
            self.coord_msg.worker_ready = self.state == "working" and self.ready_to_request
            self.coord_msg.task_complete = self.state == "owning" and self.clue_published
            self.coord_pub.publish(self.coord_msg)
            rate.sleep()

    def update_state(self):
        """
            Updates the state of the drone based on coordination msgs and whether sign is 
            adequately centered in frame.
        """
        previous_state = self.state
        if self.coord_msg.current_controller == self.WORKER:
            self.state = "owning"
            if previous_state == "working":
                # clear errors from when overseer was in control
                self.x_pid.integral = 0
                self.y_pid.integral = 0
                self.x_pid.previous_error = 0
                self.y_pid.previous_error = 0
        else:
            self.state = "working"
        return
    
    def execute_working(self):
        return
    
    def execute_centering(self):
        """
            Checks if map centroid - frame centroid error < threshold
            - if map centroid - frame centroid error < threshold for 1s, then switch self.state = "commanding"
            - always performs PID control to try to prevent drifting error due to wind
        """
        # always perform PID
        self.current_twist.linear.x = self.x_pid.update(self.error_x, 1.0 / self.UPDATE_HZ)
        self.current_twist.linear.y = self.y_pid.update(self.error_y, 1.0 / self.UPDATE_HZ)
        return
    
    def read_sign(self, cv_image):
        """
            Uses a neural network to read the sign in the image and return a confidence in the read and the predicted clue.
            For now, this is just a placeholder that returns random values. You can replace this with your actual neural network inference code.
        """
        return 0.9, "THIS"

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
        bridge = WorkerDroneNode()
        bridge.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()