from ros_robot_controller.ros_robot_controller_sdk import Board

if __name__ == "__main__":
    print("disabling torque")
    board = Board()
    for i in range(1, 24):
        board.bus_servo_enable_torque(i, 1)
    # board.bus_servo_enable_torque(1, 0)
    # board.bus_servo_set_position(0.02, [(1, 400)])