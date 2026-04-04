#!/usr/bin/env python3
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

class OverseerDroneNode:
    def __init__(self):
        rospy.init_node('overseer', anonymous=True)
        self.UPDATE_HZ = 30
        
        # Subscribe to the camera topic defined in Gazebo plugin
        self.image_sub = rospy.Subscriber("camera2/image_raw", Image, self.image_callback)
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1) # cmd_vel for cmd_bridge node to apply as wrenches
        rospy.loginfo("Overseer Down Camera Node Initialized. Waiting for images...")

        self.abs_elev_pub = rospy.Publisher("abs_z_target", Float64, queue_size=1) # publish absolute elevation target for cmd bridge to maintain for oversight
        self.alt_sub = rospy.Subscriber("altitude", Float64, self.alt_callback)
        self.abs_elev_target = 8 # desired absolute elevation target for oversight, adjust as needed
        self.altitude = 0.1 # current altitude of drone, updated from laser scan data

        # state variable to track if drone is in position for oversight
        self.bridge = CvBridge()
        self.window_name = f"Overseer masked camera feed: {rospy.get_namespace()}"

        self.current_twist = Twist()

        self.state = "init_rise"
        self.centered = False
        self.INIT_RISE_TIME = 0.25 # seconds to rise up before switching to in position state
        self.INIT_RISE_SPEED_Z = 0.9
        self.INIT_RISE_SPEED_Y = 5.5
        self.INIT_RISE_SPEED_X = self.INIT_RISE_SPEED_Y * 5.5/2.5

        self.x_pid = PIDController(kp=0.01, ki=0.0, kd=0.0)
        self.y_pid = PIDController(kp=0.01, ki=0.0, kd=0.0)
        self.error_x = 0.0
        self.error_y = 0.0
        self.consecutive_centers = 0

        rospy.sleep(0.5)
    
    def alt_callback(self, msg):
        self.altitude = msg.data

    def image_callback(self, data):
        # only process images for centering once initial rise is complete.
        if self.state != "init_rise":
            try:
                # Convert ROS Image message to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                gray_image = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
                # display image with edges for debugging
                edges = cv.Canny(gray_image, 50, 150, apertureSize=3)
                edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
                if len(contours) > 0:
                    # display largest contour for debugging
                    largest_contour = max(contours, key=cv.contourArea)

                    # bounding box of whole map
                    x, y, w, h = cv.boundingRect(largest_contour)
                    cX = x + w // 2
                    cY = y + h // 2
                    cv.drawContours(edges_bgr, [largest_contour], -1, (255,0,0), 3)
                    cv.rectangle(edges_bgr, (x, y), (x+w, y+h), (0,0,255), 2)
                    cv.circle(edges_bgr, (cX, cY), 7, (0,255,0), -1)
                    M = cv.moments(largest_contour)

                    self.error_x = cX - cv_image.shape[1]//2
                    self.error_y = cY - cv_image.shape[0]//2

                cv.imshow(self.window_name, edges_bgr)
                cv.waitKey(1)
                
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
        return

    def run(self):
        rate = rospy.Rate(self.UPDATE_HZ)
        self.start_time = rospy.get_time()
        while not rospy.is_shutdown():
            if self.state == "init_rise":
                self.execute_initial_rise()
            
            elif self.state == "centering":
                self.execute_centering()

            elif self.state == "commanding":
                self.abs_elev_target = 7.8
                self.execute_centering()

            elif self.centered:
                self.state = "in_position"

            self.vel_pub.publish(self.current_twist)
            rate.sleep()
    
    def execute_initial_rise(self):
        """
            Sends twist messages on cmd_vel topic for drone_cmd_bridge to pick up and apply wrenches to drone.
            - Calls to fly toward centre of map and up to goal height
            - Modifies self.current_twist, self.state, and uses self.vel_pub
            - Sets state = "centering" once 2 * self.INIT_RISE_TIME passes
        """
        current_time = rospy.get_time() 

        if current_time - self.start_time > 2 * self.INIT_RISE_TIME:                
            self.abs_elev_pub.publish(self.abs_elev_target) # publish an absolute elevation target for the cmd bridge to maintain for oversight
            self.current_twist.linear.y = 0
            self.current_twist.linear.x = 0
            self.current_twist.linear.z = 0

            print(f"Finished rising at height {self.altitude}m")
            print("Centering on world...")

            self.state = "centering" # set state to in position for oversight

        elif current_time - self.start_time > self.INIT_RISE_TIME:
            self.current_twist.linear.z = - 2.0 * self.INIT_RISE_SPEED_Z
            self.current_twist.linear.y = self.INIT_RISE_SPEED_Y - 0.005
            self.current_twist.linear.x = self.INIT_RISE_SPEED_X - 0.005

        else:
            # publish a twist to cmd_vel to rise up to a certain height for oversight
            self.current_twist.linear.z = self.INIT_RISE_SPEED_Z 
            self.current_twist.linear.y = - self.INIT_RISE_SPEED_Y 
            self.current_twist.linear.x = - self.INIT_RISE_SPEED_X 

        return
    
    def execute_centering(self):
        """
            Checks if map centroid - frame centroid error < threshold
            - if map centroid - frame centroid error < threshold, changes state
        """
        # always perform PID
        self.current_twist.linear.x = self.x_pid.update(self.error_x, 1.0 / self.UPDATE_HZ)
        self.current_twist.linear.y = self.y_pid.update(self.error_y, 1.0 / self.UPDATE_HZ)
            
        # check if center of contour is close enough to center of image to be considered centered
        if abs(self.error_x) < 20 and abs(self.error_y) < 20:
            self.consecutive_centers += 1
            if self.consecutive_centers > self.UPDATE_HZ:
                self.state = "commanding"
                print("Drone is centered on world, commanding...")
        else:
            self.consecutive_centres = 0
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