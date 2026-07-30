#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2023/11/10
import cmd
import os
import time
import math
import rclpy
import threading
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from rclpy.executors import MultiThreadedExecutor

# Added imports for torque management
from ros_robot_controller_msgs.msg import SetBusServoState, BusServoState

from servo_controller.servo_controller import ServoManager
from servo_controller.joint_position_controller import JointPositionController
from servo_controller_msgs.msg import ServoPosition, ServosPosition, ServoState, ServoStateList
from servo_controller.joint_trajectory_action_controller import JointTrajectoryActionController

class ControllerManager(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)  # 允许未声明的参数
        self.joints = ['coxa_LF_joint', 'femur_LF_joint', 'tibla_LF_joint', 'coxa_LM_joint', 'femur_LM_joint', 'tibla_LM_joint', 
                       'coxa_LR_joint', 'femur_LR_joint', 'tibla_LR_joint', 'coxa_RF_joint', 'femur_RF_joint', 'tibla_RF_joint', 
                       'coxa_RM_joint', 'femur_RM_joint', 'tibla_RM_joint', 'coxa_RR_joint', 'femur_RR_joint', 'tibla_RR_joint', 
                       'joint1', 'joint2', 'joint3','joint4','joint5','r_joint']
             
        self.leg_ids = [5, 3, 1, 11, 9, 7, 17, 15, 13, 18, 16, 14, 12, 10, 8, 6, 4, 2]
        self.legs_sleep_pose = [0.2471, 1.9, -1.2, 0.0, 1.9, -1.2, -0.251, 1.9, -1.2, 0.255, 1.9, -1.2, 0.0, 1.9, -1.2, -0.263, 1.9, -1.2]
        self.legs_wake_pose = [0.2471,0.816,-0.552,0.0,0.904,-0.678,-0.251,0.821,-0.4733,0.255,0.808,-0.485,0.0,0.90,-0.678,-0.263,0.8,-0.477]

        self.arm_ids = [19, 20, 21, 22, 23, 24]
        # Leaving arm wake pose unused for now because not using arm. Also the sleep pose is good enough as a starting pose
        self.arm_wake_pose = [0.0, 0.92, -1.549, -1.466, 0.0, 0.0]
        self.arm_sleep_pose = [0.0,1.365,-1.9,-1.7,0.0,0.0]
        self.servo_ids = self.leg_ids + self.arm_ids
        self.pose_duration = 1.0

        # 读取配置参数
        self.base_frame = self.get_parameter('base_frame').value
        
        # trajectory_controller的初始化
        self.controllers = {}
        connected_ids = {}
        for i in self.joints:
            joint = self.get_parameters_by_prefix(i)
            connected_ids[str(joint['id'].value)] = i
            controller = JointPositionController(joint, i)
            self.controllers[i] = controller

        # 实例化舵机管理节点
        self.servo_manager = ServoManager(connected_ids)

        for i in ['leg_controller', 'arm_controller', 'gripper_controller']:
            controller = self.get_parameters_by_prefix(i)
            controllers = [self.controllers[joint_name] for joint_name in controller['joint_controllers'].value]
            self.controllers[i] = JointTrajectoryActionController(self,self.servo_manager, i, controllers)


        self.joint_states_pub = self.create_publisher(JointState, '~/joint_states', 1)
        self.servo_states_pub = self.create_publisher(ServoStateList, '~/servo_states', 1)
        self.create_subscription(ServosPosition, 'servo_controller', self.servo_controller_callback, 1)
        self.create_subscription(JointState, 'joint_controller', self.joint_controller_callback, 1)

        self.clock = self.get_clock()
        # 确保ros_robot_controller已完成初始化 
        namespace = self.get_namespace()
        if namespace == '/':
            namespace = ''
            
        # Torque management publisher
        self.torque_enable_pub = self.create_publisher(SetBusServoState, namespace + '/ros_robot_controller/bus_servo/set_state', 1)
        
        self.client = self.create_client(Trigger, namespace + '/ros_robot_controller/init_finish')
        self.client.wait_for_service()

        threading.Thread(target=self.publish_joint_states, daemon=True).start()
        
        # Existing init service
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        
        # New Wake and Sleep services
        self.create_service(Trigger, '~/wake', self.wake_callback)
        self.create_service(Trigger, '~/sleep', self.sleep_callback)

        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')
        
        # Disable torques on startup after a brief delay
        threading.Thread(target=self.delayed_startup_sleep, daemon=True).start()

    def delayed_startup_sleep(self):
        time.sleep(5)
        self.sleep_robot()

    def _build_servo_message(self, servo_rads, duration=None):
        msg = ServosPosition()
        msg.position_unit = "rad"
        msg.duration = self.pose_duration if duration is None else duration
        for servo_id, position in zip(self.servo_ids, servo_rads):
            data = ServoPosition()
            data.id = servo_id
            data.position = position
            msg.position.append(data)
        return msg

    def _move_robot_to_pose(self, servo_rads):
        self._servo_controller(self._build_servo_message(servo_rads))

    def wake_robot(self):
        self.set_all_torques(enable=True)
        self._move_robot_to_pose(self.legs_wake_pose + self.arm_sleep_pose)

    def sleep_robot(self):
        self._move_robot_to_pose(self.legs_sleep_pose + self.arm_sleep_pose)
        time.sleep(0.5)
        self.set_all_torques(enable=False)

    def wake_callback(self, request, response):
        self.wake_robot()
        response.success = True
        response.message = "Robot awake and torques enabled."
        return response

    def sleep_callback(self, request, response):
        self.sleep_robot()
        response.success = True
        response.message = "Robot sleeping and torques disabled."
        return response

    def set_all_torques(self, enable=True):
        self.get_logger().info("Enabling All Servo Torques" if enable else "Disabling All Servo Torques")
        servo_ids = [i for i in range(1, 25)]

        msg = SetBusServoState()
        msg.duration = 0.0
        msg.state = []

        for sid in servo_ids:
            servo_state = BusServoState()
            servo_state.present_id = [1, sid]
            servo_state.enable_torque = [1, 0 if enable else 1] # 0 = Torque On
            msg.state.append(servo_state)
            
        self.torque_enable_pub.publish(msg)

    def get_node_state(self, request, response):
        response.success = True
        return response
    
    def _servo_controller(self, msg):
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

    def servo_controller_callback(self, msg):
        # self.get_logger().info('\033[1;32m%s\033[0m' % str(msg))
        self._servo_controller(msg)

    def joint_controller_callback(self, msg):
        for name, position in zip(msg.name, msg.position):
            if name in self.controllers:
                self.servo_manager.set_position(self.controllers[name].servo_id, self.controllers[name].pos_rad_to_pulse(position))
                time.sleep(0.005)

    def publish_joint_states(self):
        while True:
            msg = JointState()
            msg.header.stamp = self.clock.now().to_msg()
            msg.header.frame_id = self.base_frame
            positions = self.servo_manager.get_position()
            servos_msg = ServoStateList()
            servos_msg.header = msg.header
            for i in positions:
                msg.name.append(positions[i].name)
                msg.position.append(self.controllers[positions[i].name].pos_pulse_to_rad(positions[i].position))
                
                servo_msg = ServoState()
                servo_msg.id = int(i)
                servo_msg.position = int(positions[i].position)
                servos_msg.servo_state.append(servo_msg)
            self.joint_states_pub.publish(msg)
            self.servo_states_pub.publish(servos_msg)
            time.sleep(0.02)


def main():
    node = ControllerManager('controller_manager')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
if __name__ == "__main__":
    main()