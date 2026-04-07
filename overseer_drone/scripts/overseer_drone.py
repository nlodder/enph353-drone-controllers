#!/usr/bin/env python3
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64, String
from drone_msgs.msg import DroneMessage
import numpy as np

class OverseerDroneNode:
    def __init__(self):
        rospy.init_node('overseer', anonymous=True)
        self.UPDATE_HZ = 30
        
        self.image_sub = rospy.Subscriber("camera_down/image_raw", Image, self.image_callback)
        
        # DRONE COORDINATION
        self.coord_pub = rospy.Publisher("/drone_coordination", DroneMessage, queue_size=10)
        self.coord_sub = rospy.Subscriber("/drone_coordination", DroneMessage, self.coordination_callback) 
        self.worker_vel_pub = rospy.Publisher("/drone_a/overseer/cmd_vel", Twist, queue_size=10)
        self.coord_msg = DroneMessage()
        self.OVERSEER = 0
        self.WORKER = 1

        # COMPETTITION START/END MESSAGES
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=10)
        self.TEAM_NAME = "CYRUS"
        self.TEAM_PASS = "1111"
        self.START_MSG = f"{self.TEAM_NAME},{self.TEAM_PASS},0,aaaa"
        self.END_MSG = f"{self.TEAM_NAME},{self.TEAM_PASS},-1,aaaa"
        
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)             # cmd_vel for cmd_bridge node to apply as wrenches
        self.alt_sub = rospy.Subscriber("altitude", Float64, self.alt_callback)    # get altitude data from depth cam
        self.abs_elev_pub = rospy.Publisher("abs_z_target", Float64, queue_size=1) # publish abs elev and overseer_cmd_bridge will maintain it
        self.abs_elev_target = 6 # desired absolute elevation target for oversight, adjust as needed
        self.altitude = 6.1 # current altitude of drone, updated from laser scan data, initialized to 6.1 to prevent erratice behaviour

        self.bridge = CvBridge()
        self.window_name = f"Overseer masked camera feed: {rospy.get_namespace()}"

        # PERSISTANT TWIST - published each cycle at UPDATE_HZ
        self.current_twist = Twist()

        # INITIAL RISE OFF START
        self.state = "init_rise"
        self.INIT_RISE_TIME = 0.15 # seconds to rise up before switching to in position state
        self.INIT_RISE_SPEED_Z = 0.5
        self.INIT_RISE_SPEED_Y = 6
        self.INIT_RISE_SPEED_X = self.INIT_RISE_SPEED_Y * 5.0/2.5

        # HORIZONTAL ALIGNMENT PID
        self.x_pid = PIDController(kp=0.05, ki=0.0, kd=0.02)
        self.y_pid = PIDController(kp=0.05, ki=0.0, kd=0.02)
        self.error_x = 0.0
        self.error_y = 0.0
        self.consecutive_centers = 0

        rospy.sleep(0.5)
    
    def coordination_callback(self, msg):
        self.coord_msg = msg
        return
    
    def alt_callback(self, msg):
        self.altitude = msg.data

    def image_callback_old(self, data):
        # NOTE: This image callback was working very well for centering, consider reverting if issues
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
        # only process images for centering once initial rise is complete.
        if self.state != "init_rise":
            try:
                # convert ROS Image message to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

                # FIND OUTLINE OF MAP
                gray_image = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
                edges = cv.Canny(gray_image, 50, 150, apertureSize=3)
                edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
                if len(contours) > 0:
                    # take convex hull of contours to overcome countour paths into centre of map (walls tied to roads etc.)
                    hulls = []
                    for contour in contours:
                        hulls.append(cv.convexHull(contour))
                    largest_hull = max(hulls, key=cv.contourArea)

                    x, y, w, h = cv.boundingRect(largest_hull)
                    cX = x + w // 2
                    cY = y + h // 2
                    cv.rectangle(edges_bgr, (x, y), (x+w, y+h), (0,0,255), 2)
                    cv.circle(edges_bgr, (cX, cY), 7, (0,0,255), -1)

                    # x axis of image is y-axis in world/drone frame anc vice-versa
                    self.error_y = cv_image.shape[1]//2 - cX
                    self.error_x = cv_image.shape[0]//2 - cY

                cv.imshow(self.window_name, edges_bgr)
                cv.waitKey(1)
                
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
        return

    def image_callback(self, data):
        DEBUG = True  
        if self.state == "init_rise":
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
            
            # kernel for removing tiny noise speckles
            kernel = np.ones((3,3), np.uint8)

            # FIND THE SIGNS (Blue -> Red Dot)
            lower_blue = np.array([90, 50, 30])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv.inRange(hsv, lower_blue, upper_blue)
            blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel) # Cleanup
            
            sign_contours = cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            sign_centroids = []
            for cnt in sign_contours:
                # Area lowered: Signs at distance are tiny!
                if cv.contourArea(cnt) > 10: 
                    M = cv.moments(cnt)
                    if M["m00"] != 0:
                        sign_centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

            # FIND THE ORANGE DRONE (Orange -> Blue Dot)
            lower_orange = np.array([12, 180, 120])
            upper_orange = np.array([18, 255, 255])
            orange_mask = cv.inRange(hsv, lower_orange, upper_orange)
            
            orange_contours = cv.findContours(orange_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            drone_centroid = None
            if orange_contours:
                largest_orange = max(orange_contours, key=cv.contourArea)
                if cv.contourArea(largest_orange) > 5:
                    M = cv.moments(largest_orange)
                    if M["m00"] != 0:
                        drone_centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # MAP CENTERING
            gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray, 50, 150)
            map_contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            
            if map_contours:
                all_pts = np.concatenate(map_contours)
                hull = cv.convexHull(all_pts)
                x, y, w, h = cv.boundingRect(hull)
                cX, cY = x + w // 2, y + h // 2

                img_cent_x = cv_image.shape[1]//2
                img_cent_y = cv_image.shape[0]//2
                map_error_x = img_cent_y - cY
                map_error_y = img_cent_x - cX

                # ADJUST ERRORS FOR PID
                alpha = 0.2 # degree to which overseer follows 
                if drone_centroid:
                    worker_x, worker_y = drone_centroid
                    error_follow_x = img_cent_y - worker_y
                    error_follow_y = img_cent_x - worker_x
                    
                    # apply blended dominance, only allow worker to contribute in +-X
                    self.error_y = map_error_y
                    self.error_x = ((1 - alpha) * map_error_x) + (alpha * error_follow_x)
                else:
                    # fallback to 100% Map Centering if worker disappears
                    self.error_y = map_error_y
                    self.error_x = map_error_x

            if DEBUG:
                # draw on original image
                for pt in sign_centroids:
                    cv.circle(cv_image, pt, 6, (0, 0, 255), -1) # Red dot on signs
                if drone_centroid:
                    cv.circle(cv_image, drone_centroid, 10, (255, 0, 0), -1) # Blue dot on drone

                if map_contours:
                    cv.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv.circle(cv_image, (cX, cY), 7, (0, 255, 0), -1)

                cv.imshow(self.window_name, cv_image)
                # cv.imshow("Blue Mask Debug", blue_mask)
                cv.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Callback Error: {e}")

    def run(self):
        """
            Runs state machine for overseer
            - state = init_rise  -> fly up and towards centre
            - state = centering  -> altitude PID and xy PID, next state if centered for 1s
            - state = commanding -> commanding worker
        """
        rate = rospy.Rate(self.UPDATE_HZ)
        self.start_time = rospy.get_time()
        self.score_pub.publish(self.START_MSG)
        self.coord_msg.current_controller = self.WORKER # until overseer is stabilized
        self.coord_msg.active_sign = 1                  # we start focusing on the first sign
        
        while not rospy.is_shutdown():
            if self.state == "init_rise":
                self.execute_initial_rise()
            
            elif self.state == "centering":
                self.execute_centering()

            elif self.state == "commanding":
                self.execute_centering()
            
            elif self.state == "end_comp":
                self.score_pub.publish(self.END_MSG)
                rospy.signal_shutdown("Competition finished")

            self.vel_pub.publish(self.current_twist)
            self.coord_pub.publish(self.coord_msg)
            rate.sleep()
    
    def execute_initial_rise(self):
        """
            Sends twist messages on cmd_vel topic for drone_cmd_bridge to pick up and apply wrenches to drone.
            - Calls to fly toward centre of map and up to goal height
            - Modifies self.current_twist, self.state, and uses self.vel_pub
            - Sets state = "centering" once 2 * self.INIT_RISE_TIME passes
        """
        self.current_twist.linear.z = self.INIT_RISE_SPEED_Z 
        self.current_twist.linear.y = - self.INIT_RISE_SPEED_Y 
        self.current_twist.linear.x = - self.INIT_RISE_SPEED_X 
        if self.altitude >= self.abs_elev_target / 2:
            self.state = "centering"
            # publish an absolute elevation target for the cmd bridge to maintain for oversight
            self.abs_elev_pub.publish(self.abs_elev_target)

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
            
        # check if center of contour is close enough to center of image to be considered centered
        if abs(self.error_x) < 20 and abs(self.error_y) < 20:
            self.consecutive_centers += 1
            if self.consecutive_centers > self.UPDATE_HZ:
                self.state = "commanding"
                # print("Drone is centered on world, commanding...")
        else:
            self.consecutive_centers = 0
            self.state = "centering"
        
        return

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
        Simple PID controller to stabilize the drone's x,y position.
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
        bridge = OverseerDroneNode()
        bridge.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()