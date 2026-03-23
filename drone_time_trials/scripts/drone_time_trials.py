#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class DroneTimeTrialNode:
    def __init__(self):
        rospy.init_node('drone_time_trials', anonymous=True)
        self.START_MSG = "CYRUS,1111,0,aaaa"
        self.END_MSG = "CYRUS,1111,-1,aaaa"
        self.SPEED = 10
        
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=10)
        rospy.loginfo("Score publisher node initialized")
        # publishing to same topic that drone_cmd_bridge.py subscribes to
        self.mov_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        rospy.loginfo("Time trials move publisher node initialized")

        self.current_y = 0.0
        self.start_y = None

        rospy.Subscriber("odom", Odometry, self.odom_callback)

        rospy.sleep(1.0) # sleep for 1s to let nodes initialize
    
    def odom_callback(self,msg):
        self.current_y = msg.pose.pose.position.y

    def run(self):
        while not rospy.is_shutdown():
            rospy.loginfo("Starting Time Trial...")
            self.score_pub.publish(self.START_MSG)

            # record starting position
            while self.start_y is None and not rospy.is_shutdown():
                self.start_y = self.current_y
                rospy.sleep(0.1)

            twist = Twist()
            twist.linear.y = self.SPEED

            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                distance_travelled = abs(self.current_y - self.start_y)

                if distance_travelled >= 1.0:
                    break
                
                self.mov_pub.publish(twist)
                rate.sleep()
            
            # stop
            self.mov_pub.publish(Twist()) # empty Twist
            self.score_pub.publish(self.END_MSG)
            rospy.loginfo("Time Trial Finished!")

            rospy.loginfo("Shutting down Time Trial node...")
            rospy.signal_shutdown("Time Trial Complete!")

if __name__ == '__main__':
    try:
        tt_node = DroneTimeTrialNode()
        tt_node.run()
    except rospy.ROSInterruptException:
        pass