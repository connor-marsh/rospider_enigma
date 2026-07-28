#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2023/11/10
import os
import sys
import time
import math
import rclpy
import threading
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from rclpy.executors import MultiThreadedExecutor
from servo_controller.servo_controller import ServoManager
from servo_controller.joint_position_controller import JointPositionController
from servo_controller_msgs.msg import ServosPosition, ServoState, ServoStateList
from servo_controller.joint_trajectory_action_controller import JointTrajectoryActionController
import numpy as np

class GaitExtractor(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)  # 允许未声明的参数
        self.joints = ['coxa_LF_joint', 'femur_LF_joint', 'tibla_LF_joint', 'coxa_LM_joint', 'femur_LM_joint', 'tibla_LM_joint', 
                       'coxa_LR_joint', 'femur_LR_joint', 'tibla_LR_joint', 'coxa_RF_joint', 'femur_RF_joint', 'tibla_RF_joint', 
                       'coxa_RM_joint', 'femur_RM_joint', 'tibla_RM_joint', 'coxa_RR_joint', 'femur_RR_joint', 'tibla_RR_joint', 
                       'joint1', 'joint2', 'joint3','joint4','joint5','r_joint']       

        self.create_subscription(ServosPosition, 'servo_controller', self.servo_controller_callback, 1)
        self.gait = []


    def servo_controller_callback(self, msg):
        # self.get_logger().info('\033[1;32m%s\033[0m' % str(msg))
        print("###########################")
        positions = []
        indices = []
        for servo in msg.position:
            positions.append(servo.position)
            indices.append(servo.id)
        print(positions)
        print(indices)
        self.gait.append(positions)
        return
        data = ServosPosition()
        positions = self.servo_manager.get_position()
        if msg.position_unit == 'pulse':
            for i in msg.position:
                if str(i.id) in positions:
                    data.position.append(i)
            self.servo_manager.set_position(msg.duration, data.position)
        elif msg.position_unit == 'rad':
            for i in msg.position:
                if str(i.id) in positions:
                    i.position = self.controllers[positions[str(i.id)].name].pos_rad_to_pulse(i.position)
                    data.position.append(i)
            self.servo_manager.set_position(msg.duration, data.position)
        elif msg.position_unit == 'deg':
            for i in msg.position:
                if str(i.id) in positions:
                    i.position = self.controllers[positions[str(i.id)].name].pos_rad_to_pulse(math.radians(i.position))
                    data.position.append(i)
            self.servo_manager.set_position(msg.duration, data.position)


def main():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_file_dir)
    print(os.getcwd())

    node = GaitExtractor('gait_extractor')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Detected CTRL+C")
    finally:
        print("Extracted movements")
        print(node.gait)
        #TODO ask for user input
        if len(sys.argv) == 2:
            
            toSave = np.array(node.gait)
            filename = sys.argv[1] + '.csv'
            np.savetxt(filename, toSave, delimiter=',')
        node.destroy_node()
if __name__ == "__main__":
    main()
