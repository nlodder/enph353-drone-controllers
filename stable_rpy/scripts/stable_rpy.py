#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Imu
import tf
import math

class DroneRPYStabilizer:
    def __init__(self):
        """
            Initializes the DroneRPYStabilizer node, sets up PID controllers for stabilization, and subscribes to necessary topics.
            - Initializes ROS node and sets up parameters for the PID controllers
            - Subscribes to "imu" for orientation data
        """
        rospy.init_node('drone_rpy_stabilizer')
        self.UPDATE_HZ = 30
        
        # PID for angular stabilization
        self.pitch_PID = StabilizePIDController(kp=0.2, ki=0.0, kd=0.02)
        self.roll_PID = StabilizePIDController(kp=0.2, ki=0.0, kd=0.02)
        self.target_euler = [0, 0] # target rp in rad
        self.TORQUE_SCALE = 2

        self.rpy_stable_pub = rospy.Publisher("rpy_stable_wrench", Wrench, queue_size=1)
        self.current_wrench = Wrench()
        self.imu_msg = None
        self.DUR_BUFFER = rospy.Duration(1.5 / self.UPDATE_HZ) # force applied for update period

        # subscribe to drone's imu topic
        rospy.Subscriber("imu", Imu, self.imu_callback)
    
    def imu_callback(self, msg):
        """
            Updates self.imu_msg
        """
        self.imu_msg = msg
        

    def run(self):
        """
            Main loop to continuously publish required torques to stabilize drone in rostopic
            - Uses a ROS rate to control the update frequency.
            - Catches any ROS exceptions and logs them.
            - Modifies self.current_wrench.torque
        """
        rate = rospy.Rate(self.UPDATE_HZ)
        # ensure we're communicating with imu
        while not rospy.is_shutdown() and self.imu_msg is None:
            rate.sleep()

        while not rospy.is_shutdown():
            self.update_torques()
            self.rpy_stable_pub.publish(self.current_wrench)
            rate.sleep()
    
    def update_torques(self):
        """
        Updates torques required to stabilize drone with PID control based on imu senor's
        most recently recieved message by mapping global Euler angle errors to body-fixed
        angular velocity damping.
        """
        msg = self.imu_msg
        quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)

        dt = 1.0/self.UPDATE_HZ

        # Pass the IMU's angular velocity directly into PID
        torque_roll = self.roll_PID.update(
            self.target_euler[0], roll, dt, current_velocity=msg.angular_velocity.x
        )
        torque_pitch = self.pitch_PID.update(
            self.target_euler[1], pitch, dt, current_velocity=msg.angular_velocity.y
        )

        self.current_wrench.torque.x = self.TORQUE_SCALE * torque_roll
        self.current_wrench.torque.y = self.TORQUE_SCALE * torque_pitch

class StabilizePIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
    
    def update(self, target_rad, current_rad, dt, current_velocity=None):
        error = target_rad - current_rad
        # normalize error [-pi, pi]
        error = math.atan2(math.sin(error), math.cos(error))
        
        # integral term
        self.integral += error * dt
        
        # derivative term: Use IMU velocity if provided, otherwise calculate
        if current_velocity is not None:
            # We use negative velocity because the derivative of error (0 - current) 
            # is -velocity. This provides the damping.
            derivative = -current_velocity
        else:
            derivative = (error - self.previous_error) / dt if dt > 0 else 0

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # keep state updated
        self.previous_error = error
        return output

if __name__ == '__main__':
    try:
        bridge = DroneRPYStabilizer()
        bridge.run()
    except rospy.ROSInterruptException:
        pass
