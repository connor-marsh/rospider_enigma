#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2023/11/10
from servo_controller_msgs.msg import ServoPosition, ServosPosition


def set_servo_position(pub, duration, positions):
    msg = ServosPosition()
    msg.duration = float(duration)
    new_position_list = []
    for i in positions:
        position = ServoPosition()
        position.id = i[0]
        position.position = float(i[1])
        new_position_list.append(position)
    msg.position = new_position_list
    msg.position_unit = "rad"
    pub.publish(msg)

if __name__ == '__main__':
    import time
    import rclpy
    from rclpy.node import Node
    import numpy as np

    # wake_pose_vals = [0.2471,0.816,-0.552,0.0,0.904,-0.678,-0.251,0.821,-0.4733,0.255,0.808,-0.485,0.0,0.90,-0.678,-0.263,0.8,-0.477,0.0,0.92,-1.549,-1.466,0.0,0.0]
    # rest_pose_vals = [0.2471, 1.9, -1.2, 0.0, 1.9, -1.2, -0.251, 1.9, -1.2, 0.255, 1.9, -1.2, 0.0, 1.9, -1.2, -0.263, 1.9, -1.2]
    # coxa_vals = wake_pose_vals[0::3]
    # femur_vals = wake_pose_vals[1::3]
    # tibla_vals = wake_pose_vals[2::3]

    servo_ids = [5, 3, 1, 11, 9, 7, 17, 15, 13, 18, 16, 14, 12, 10, 8, 6, 4, 2]
    arm_ids = [19, 20, 21, 22, 23, 24]
    coxa_vals = [0.2471, 0.0, -0.251, 0.255, 0.0, -0.263]
    femur_vals = [0.816, 0.904, 0.821, 0.808, 0.9, 0.8]
    femur_vals = [1.9]*6
    tibla_vals = [-0.552, -0.678, -0.4733, -0.485, -0.678, -0.477]
    tibla_vals = [-1.2]*6
    coxa_ids = servo_ids[0::3]
    femur_ids = servo_ids[1::3]
    tibla_ids = servo_ids[2::3]
    print(coxa_vals)
    print(femur_vals)
    print(tibla_vals)
    printout = []
    for i in range(6):
        printout.append(coxa_vals[i])
        printout.append(femur_vals[i])
        printout.append(tibla_vals[i])
    print(printout)

    coxa_commands = [(coxa_ids[i], coxa_vals[i]) for i in range(len(coxa_ids))]
    femur_commands = [(femur_ids[i], femur_vals[i]) for i in range(len(femur_ids))]
    tibla_commands = [(tibla_ids[i], tibla_vals[i]) for i in range(len(tibla_ids))]
    commands = coxa_commands+femur_commands+tibla_commands

    # all_vals = coxa_vals+femur_vals+tibla_vals
    # commands = [(servo_ids[i], wake_pose_vals[i]) for i in range(len(servo_ids))]

    rclpy.init()
    node = Node('servo_control_demo')
    pub = node.create_publisher(ServosPosition, 'servo_controller', 1)
    time.sleep(1)
    set_servo_position(pub, 0.02, commands)

    # while rclpy.ok():
    #     for i in range(1,2):
    #         set_servo_position(pub, 0.02, [(i, 50)])
    #         time.sleep(1)



