#!/usr/bin/env python3
import rclpy
import time
# Replace 'your_package_name' with the actual package name
from ros_robot_controller_msgs.msg import SetBusServoState, BusServoState

def disable_all_torque(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('torque_manager')
    
    # Replace '/your_topic_name' with the actual topic
    pub = node.create_publisher(SetBusServoState, '/your_topic_name', 10)
    
    # Allow a brief moment for the DDS network to discover the subscriber
    time.sleep(0.5) 

    msg = SetBusServoState()
    msg.duration = 0.0
    msg.state = []

    # Loop through IDs 1 to 24 inclusive, appending to the single message array
    for sid in range(1, 25):
        servo_state = BusServoState()
        servo_state.present_id = [1, sid]
        servo_state.enable_torque = [1, 0] # 0 = Torque Off
        msg.state.append(servo_state)

    node.get_logger().info("Publishing single message to disable torque for Servos 1-24...")
    pub.publish(msg)
    
    # Give the DDS layer a fraction of a second to actually push the data before destroying the node
    time.sleep(0.1)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    disable_all_torque()