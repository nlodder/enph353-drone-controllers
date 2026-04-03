#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Imu
from gazebo_msgs.srv import ApplyBodyWrench

class Stabilizer:
    def __init__(self):
        rospy.init_node('stabilizer')
        self.ns = rospy.get_namespace().strip('/')
        self.UPDATE_HZ = 30
        self.P = 0.1

        # latch last command
        self.current_wrench = Wrench()

        # subscribe to drone's imu topic
        rospy.Subscriber("imu", Imu, self.imu_callback)
        # these will become dronex/planar_cmd and dronex/vert_cmd from launch namespacing
        # self.ang_pub = rospy.Publisher("ang_force", Wrench, queue_size=1)
    
    def imu_callback(self, msg):
        # simple P controller to stabilize drone's orientation
        # this is a placeholder for your actual control algorithm
        self.current_wrench.torque.x = -self.P * msg.angular_velocity.x
        self.current_wrench.torque.y = -self.P * msg.angular_velocity.y
        self.current_wrench.torque.z = -self.P * msg.angular_velocity.z
        print(f"IMU data: {msg.angular_velocity}, Published wrench: {self.current_wrench}")
    
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        try:
            apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
            apply_wrench(body_name=f"{self.ns}::link_drone_body", wrench=self.current_wrench, duration=rospy.Duration(0.1))
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

if __name__ == '__main__':
    rospy.init_node('stabilizer')
    stabilizer = Stabilizer()
    rospy.spin()