#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Imu
import tf
import math

class DroneRPStabilizer:
    def __init__(self):
        """
            Initializes the DroneRPStabilizer node, sets up PID controllers for roll/pitch stabilization, and subscribes to necessary topics.
            - Initializes ROS node and sets up parameters for the PID controllers for roll and pitch
            - Subscribes to "imu" for orientation data
            - Prepares to apply twists to the drone in Gazebo based on the received commands and sensor data.
        """
        rospy.init_node('drone_rp_stabilizer')
        
        # PID for angular stabilization
        self.pitch_PID = StabilizePIDController(kp=0.1, ki=0.0, kd=0.02)
        self.roll_PID = StabilizePIDController(kp=0.1, ki=0.0, kd=0.02)
        self.target_euler = [0, 0] # target roll and pitch in radians

        self.UPDATE_HZ = 30
        self.DUR_BUFFER = rospy.Duration(1.5 / self.UPDATE_HZ) # force applied for update period

        self.current_wrench = Wrench()

        # subscribe to drone's imu topic
        rospy.Subscriber("imu", Imu, self.imu_callback)
        # publish roll and pitch stabilization torques for the CMD bridge to apply as wrenches
        self.orientation_stabilizer_pub = rospy.Publisher("rp_stabilizer_wrench", Wrench, queue_size=1) # for debugging, can visualize in rqt_plot
        
    def imu_callback(self, msg):
        """Updates the current orientation of the drone and computes the necessary torques to stabilize roll and pitch angles."""
        # get orientation from the IMU message as a quaternion
        # this is the orientation of the drone in the world frame
        quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion) # convert to radians

        torque_roll = self.roll_PID.update(self.target_euler[0], roll, 1.0 / self.UPDATE_HZ)
        torque_pitch = self.pitch_PID.update(self.target_euler[1], pitch, 1.0 / self.UPDATE_HZ)

        # PD for angular stabilization
        # set to 0 for now...
        self.current_wrench.torque.x = 0 # torque_roll
        self.current_wrench.torque.y = 0 # torque_pitch
        self.current_wrench.torque.z = 0  # No stabilization needed for yaw
        # self.show_pattern(quaternion, euler, error_roll, error_pitch)

    def run(self):
        """
            Main loop to continuously publish required torques to stabilize drone in rostopic
            - Uses a ROS rate to control the update frequency.
            - Catches any ROS exceptions and logs them.
        """
        rate = rospy.Rate(self.UPDATE_HZ)
        while not rospy.is_shutdown():
            try:
                self.orientation_stabilizer_pub.publish(self.current_wrench) # publish the current wrench for debugging/visualization
            except rospy.ROSException as e:
                rospy.logerr(f"Failed to publish stabilizer wrench: {e}")

            rate.sleep()
    
    def show_pattern(self, quat, euler, error_roll, error_pitch):
        """
            Utility function to print the current orientation and stabilization errors in a readable format directly in terminal.
        """
        pattern = f"Quat: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}] | "
        pattern += f"Euler: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] | "
        pattern += f"Error Roll: {error_roll:.2f} | Error Pitch: {error_pitch:.2f}"
        rospy.loginfo(pattern)

class StabilizePIDController:
    """Simple PID controller for stabilizing roll and pitch angles."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
    
    def update(self, target_rad, current_rad, dt):
        """
        Simple PID controller to stabilize the drone's orientation.
         - target_rad: desired angle in radians (roll or pitch)
         - current_rad: current angle in radians (roll or pitch)
         - dt: time step in seconds
         Returns the control output (torque) to apply.
        """
        # if the roll/pitch is positive, apply negative torque to stabilize, and vice versa
        error = target_rad - current_rad
        # normalize error to be within [-pi, pi]
        error = math.atan2(math.sin(error), math.cos(error))
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

if __name__ == '__main__':
    try:
        bridge = DroneRPStabilizer()
        bridge.run()
    except rospy.ROSInterruptException:
        pass
