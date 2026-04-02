#!/usr/bin/env python3
from drone_desc.scripts.drone_cmd_bridge import DroneCmdBridge
import rospy
from geometry_msgs.msg import Wrench, Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

class Stabilizer:
    def __init__(self):
        rospy.init_node('stabilizer')
        self.UPDATE_HZ = 60

        # latch last command
        self.current_wrench = Wrench()

        # subscribe to drone's imu topic
        rospy.Subscriber("imu", Imu, self.imu_callback)
        # these will become dronex/planar_cmd and dronex/vert_cmd from launch namespacing
        self.ang_pub = rospy.Publisher("ang_vel", Twist, queue_size=1)
    
    def imu_callback(self, msg):
        self.current_wrench = msg
    
    def run(self):
        # rate of cmd updates to planar and vertical movement plugins
        rate = rospy.Rate(self.UPDATE_HZ)
        while not rospy.is_shutdown():
            self.mov_pub.publish(self.current_wrench)
            rate.sleep()

if __name__ == '__main__':
    try:
        bridge = DroneCmdBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        pass