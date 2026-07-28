from ros_robot_controller.ros_robot_controller_sdk import Board

if __name__ == "__main__":
    # print("disabling torque")
    board = Board()
    # for i in range(1, 24):
    #     board.bus_servo_enable_torque(i, 1)
    # board.bus_servo_enable_torque(1, 0)
    servo_ids = [5, 3, 1, 11, 9, 7, 17, 15, 13, 18, 16, 14, 12, 10, 8, 6, 4, 2]
    coxa_ids = servo_ids[0::3]
    femur_ids = servo_ids[1::3]
    tibla_ids = servo_ids[2::3]
    coxa_value = 500
    femur_value = 75
    tibla_value = 760
    
    coxa_commands = [(coxa_ids[i], coxa_value) if i < 3 else (coxa_ids[i], 1000-coxa_value) for i in range(len(coxa_ids))]
    femur_commands = [(femur_ids[i], femur_value) if i < 3 else (femur_ids[i], 1000-femur_value) for i in range(len(femur_ids))]
    tibla_commands = [(tibla_ids[i], tibla_value) if i < 3 else (tibla_ids[i], 1000-tibla_value) for i in range(len(tibla_ids))]
    commands = coxa_commands+femur_commands+tibla_commands
    print(commands)
    # board.bus_servo_set_position(0.02, commands)

    rest_pose = [(5, 500), (11, 500), (17, 500), (18, 500), (12, 500), (6, 500), (3, 75), (9, 75), (15, 75), (16, 925), (10, 925), (4, 925), (1, 760), (7, 760), (13, 760), (14, 240), (8, 240), (2, 240)]

    board.bus_servo_set_position(0.02, [(3, 500)])

    
    # print(board.bus_servo_read_position(1))