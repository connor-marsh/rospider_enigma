#!/usr/bin/env python3
# encoding: utf-8
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Bool
from ros_robot_controller_msgs.msg import SetBusServoState, BusServoState
import time

class TorqueManager(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)  # 允许未声明的参数(allow undeclared parameters)
        
        namespace = self.get_namespace()
        if namespace == '/':
            namespace = ''
       
        self.torque_enable_pub = self.create_publisher(SetBusServoState, '/ros_robot_controller/bus_servo/set_state', 1)
        self.create_subscription(Bool, 'set_all_torques', self.set_all_torques_callback, 1)

        self.client = self.create_client(Trigger, namespace + '/controller_manager/init_finish')
        self.client.wait_for_service()
        # Wait 5 seconds then disable all torques on startup
        time.sleep(5)
        self.set_all_torques(False)


        
    
    def set_all_torques_callback(self, msg):
        self.set_all_torques(enable=msg.data)

    def set_all_torques(self, enable=True):
        print("Enabling" if enable else "Disabling", "All Servo Torques")

        servo_ids = [i for i in range(1,25)]

        msg = SetBusServoState()
        msg.duration = 0.0
        msg.state = []

        for sid in servo_ids:
            servo_state = BusServoState()
            servo_state.present_id = [1, sid]
            servo_state.enable_torque = [1, 0 if enable else 1] # 0 = Torque On
            msg.state.append(servo_state)
        self.torque_enable_pub.publish(msg)

def main():
    node = TorqueManager('torque_manager')
    rclpy.spin(node)  # 循环等待ROS2退出(loop waiting for ROS2 to exit)

if __name__ == "__main__":
    main()
